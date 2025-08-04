"""

Bewerberprozess AIS AI-Team
(2) EP-Zuordnung
Noah Wolfahrt, 02.08.2025

Dieses Skript weist einer Anlage, die textlich beschrieben wird,
eine Anlagennummer aus dem EP-Katalog zu.

"""


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

rng = np.random.RandomState(146)



#-----------------------------------------------
# Schritt 1: Daten importieren und aufbereiten
#-----------------------------------------------

# Datei EP-Zuordnung importieren
df_allocation = pd.read_csv("EP_Katalog.csv")
df_allocation = df_allocation.iloc[:, -2:]
df_allocation = df_allocation.rename(
    columns={"Kurztext / Bezeichnung": "Name", "Artikelnummer": "ID"})
df_allocation["Name"] = df_allocation["Name"].str[6:]   # ID aus Name löschen
df_allocation = df_allocation.dropna(axis=0)
df_allocation = df_allocation[df_allocation["ID"] != "ohne"]


# Datei Beispielobjekte importieren
df_example = pd.read_csv("Beispielobjekte_Anlagen.csv")
df_example = df_example[["Anlagenname", "AKS-Bezeichnung", "Kostengruppenbezeichnung", "Merkmale", "Verbandsnummer"]]
df_example = df_example.rename(columns={"Anlagenname": "Name", "Verbandsnummer": "ID"})
df_example = df_example[~(df_example["ID"].isna())]
df_example = df_example[~(df_example["ID"] == "ohne")]


# Daten für weitere Berechnungen transformieren
# Es werden alle relevanten Texte zu einer großen Spalte zusammengefasst.
df_example["Name"] = df_example[
    ["Name", "AKS-Bezeichnung", "Kostengruppenbezeichnung", "Merkmale"]
    ].astype(str).agg(" ".join, axis=1)
df_example = df_example[["Name", "ID"]]


# Daten aggregieren
full_dataset = pd.concat([df_example, df_allocation], axis=0)


# Dictionary aller möglichen IDs gruppiert nach Normalgruppe erstellen
anlagen_dict = {}
for nummer in full_dataset["ID"]:
    key = nummer[:3]
    anlagen_dict[f"{key}"] = set()
for nummer in full_dataset["ID"]:
    key = nummer[:3]
    anlagen_dict[f"{key}"].add(nummer)


# Training und Test Set erstellen
X_train, X_test, y_train, y_test = train_test_split(
    full_dataset["Name"], full_dataset["ID"],
    test_size=0.2, random_state=rng)



#-----------------------------------------------
# Schritt 2: Normalgruppen vorhersagen
#-----------------------------------------------

# Beschreibung und IDs für Modell konvertieren
# Diese Transformer dürfen das ganze Dataset sehen!
# Es geht hier nur um Umbeschriftung, daher müssen alle Labels miteinbezogen werden.
tfidf = TfidfVectorizer()
tfidf.fit(full_dataset["Name"])
label_encoder = LabelEncoder()
label_encoder.fit(full_dataset["ID"].str[:3])


# Datensatz transformieren
X_train_tf = tfidf.transform(X_train)
X_test_tf = tfidf.transform(X_test)
y_train_tf = label_encoder.transform(y_train.str[:3])
y_test_tf = label_encoder.transform(y_test.str[:3])


# Modell für Normalgruppe erstellen und trainieren, Labels vorhersagen
# Diese Transformer dürfen nicht das ganze Dataset sehen!
# Sie lernen auf Grundlage der Trainingsdaten und sollen später neue Testdaten sehen.
logreg = LogisticRegression(random_state=rng)
logreg.fit(X_train_tf, y_train_tf)
y_pred = logreg.predict(X_test_tf)
y_pred = pd.Series(label_encoder.inverse_transform(y_pred),
                   index=X_test.index, name="Pred NG")


# Genauigkeitsscore berechnen
accuracy = accuracy_score(y_test.str[:3], y_pred)
print(f"\nGenauigkeit der Normalgruppen-Vorhersage: {np.round(accuracy * 100, 2)} %")



#-----------------------------------------------
# Schritt 3: Modelltraining für Normalgruppen
#-----------------------------------------------

# Trainiere für jede Normalgruppe ein separates Modell auf die zugehörigen Normalpositionen.

# Dazu braucht es eine neue Umbeschriftung für die (jetzt) vollständigen IDs.
# Die Vektorisierung der Bezeichnungen mit TFIDF kann aber beibehalten werden.
label_encoder_full = LabelEncoder()
label_encoder_full.fit(full_dataset["ID"])


# Die Gesamtheit aller Modelle wird in einem Dictionary gespeichert.
model_dict = {}

# Trainiere die Modelle und lege sie im Dictionary ab.
for key in anlagen_dict.keys():
    # Gibt es nur eine Normalposition, ist nichts zu tun.
    if len(anlagen_dict[f"{key}"]) > 1:

        # Initialisiere ein neues Modell für jede Normalgruppe.
        clf = DecisionTreeClassifier(random_state=rng)

        # Trainiere nur auf Daten, die zur jeweiligen Normalgruppe passen.
        # Transformiere die Daten vorher für das Modell.
        pos_data = full_dataset[full_dataset["ID"].str[:3] == key]
        X_pos = tfidf.transform(pos_data["Name"])
        y_pos = label_encoder_full.transform(pos_data["ID"])

        # Training und Test Set für Modell erstellen
        X_pos_train, X_pos_test, y_pos_train, y_pos_test = train_test_split(
            X_pos, y_pos, test_size=0.2, random_state=rng)

        # Modell trainieren und abspeichern
        clf.fit(X_pos_train, y_pos_train)
        model_dict[f"model{key}"] = clf



#-----------------------------------------------
# Schritt 4: Normalposition auf Grundlage der Normalgruppe vorhersagen
#-----------------------------------------------

# Gegeben sein sollen die Beschreibungen einer Normalposition
# sowie ihre vorhergesagte Normalgruppe - siehe Schritt 2.
full_testset = pd.concat([X_test, y_pred], axis=1)


# Auf Grundlage der Normalgruppe soll nun das richtige Modell verwendet werden,
# um die Normalposition vorherzusagen.
prediction = pd.Series(0, index=X_test.index, name="Pred NP")


# Befüllung der Vorhersage-Series:
def predict_id(row):
    """Berechnet die Normalposition einer Anlage.

    Weist einer Kombination aus Text und Normalgruppe die entsprechende
    Anlagennummer zu.

    Args:
        row (pd.Series: [str, str])

    Returns:
        str: Berechnete Normalposition.
        
    """
    gruppe = row["Pred NG"]

    # Gibt es nur eine Normalposition, gib diese aus:
    if len(anlagen_dict[f"{gruppe}"]) == 1:
        return list(anlagen_dict[f"{gruppe}"])[0]

    # Ansonsten wähle das entsprechende Modell aus und sage vorher:
    clf = model_dict[f"model{gruppe}"]
    prediction_tf = clf.predict(tfidf.transform([row["Name"]]))
    prediction_utf = label_encoder_full.inverse_transform(prediction_tf)
    return prediction_utf[0]


prediction = pd.Series(full_testset.apply(predict_id, axis=1),
                       index=X_test.index, name="Pred ID")


# Aggregation der Ergebnisse
full_testset = pd.concat([full_testset, y_test, prediction], axis=1)
full_testset = full_testset[["Name", "ID", "Pred ID"]]


# Genauigkeitsscore berechnen
accuracy_id = accuracy_score(full_testset["ID"], full_testset["Pred ID"])
print(f"Genauigkeit der Normalpositionen-Vorhersage: {np.round(accuracy_id * 100, 2)} %")



#-----------------------------------------------
# Schritt 5: Anlagen der Kundendatei zuordnen
#-----------------------------------------------

# Kundendatei importieren und aufbereiten
kunde = pd.read_csv("Kundendatei.csv")
kunde["Name"] = kunde[
    ["EQ-Bezeichnung", "EQ-Klasse-Bezeichnung", "Gewerk", "Anlagenausprägung"]
    ].astype(str).agg(" ".join, axis=1)
kunde = kunde.set_index(kunde["WirtEinh"])
kunde = kunde["Name"]


# Anlagen jeweilige Anlagen-ID zuordnen

# Schritt 1: Normalgruppe zuweisen
kunde_tf = tfidf.transform(kunde)
kunde_pred_ng = logreg.predict(kunde_tf)
kunde_pred_ng = pd.Series(label_encoder.inverse_transform(kunde_pred_ng),
                   index=kunde.index, name="Pred NG")
kunde_pred_ng_full = pd.concat([kunde, kunde_pred_ng], axis=1)

# Schritt 2: Normalposition zuweisen
kunde_pred_np = pd.Series(0, index=kunde_pred_ng_full.index, name="Pred NP")
kunde_pred_np = pd.Series(kunde_pred_ng_full.apply(predict_id, axis=1),
                       index=kunde_pred_ng_full.index, name="Pred ID")
kunde_pred_np_full = pd.concat([kunde, kunde_pred_np], axis=1)

# Entkommentieren, um Datei zu exportieren
# kunde_pred_np_full.to_csv("Kundendatei Anlagenzuordnung.csv")
