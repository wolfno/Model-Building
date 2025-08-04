"""

Bewerberprozess AIS AI-Team
(1) Vollständigkeitsprüfung
Noah Wolfahrt, 02.08.2025

Dieses Skript überprüft alle Anlagen eines Gebäudes auf Vollständigkeit.

"""

import numpy as np
import pandas as pd
from warnings import simplefilter
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

rng = np.random.RandomState(146)

# Ignoriere Performance Warnings von Pandas
# Die Pivot-Matrizen sind sparse, das ist aber nicht relevant.
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# Datei Beispielobjekte importieren und formatieren
df = pd.read_csv("Beispielobjekte_Anlagen.csv")
df = df.drop(df[(df["Anlagentyp"] == "Bauteil") |
                 df["Verbandsnummer"].isna()
               ].index)
df = df[["Gebäude-ID", "Verbandsnummer"]].astype("object")
df = df.drop_duplicates()


# Daten zur weiteren Berechnung pivotieren
df_pivot = pd.crosstab(df["Gebäude-ID"], df["Verbandsnummer"])
df_pivot = df_pivot.drop(columns="ohne")


# Mittels EP-Katalog sicherstellen, dass alle Anlagen gespeichert sind
katalog = pd.read_csv("EP_Katalog.csv")
katalog = katalog["Artikelnummer"].astype(str)
katalog = katalog.drop_duplicates()
for anlagennummer in katalog:
    if anlagennummer != np.nan and anlagennummer not in df_pivot.columns:
        df_pivot[anlagennummer] = 0


# Training und Test Set erstellen
X_train, X_test = train_test_split(
    df_pivot, test_size=0.2, random_state=rng)



#-----------------------------------------------
# Clustering der Gebäude aus dem Training Set mit KMeans
#-----------------------------------------------

# Training Set clustern
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=rng)
X_train_clusterID = pd.DataFrame(
                kmeans.fit_predict(X_train),
                columns=["ClusterID"], index=X_train.index)
X_train_clustered = pd.concat([X_train, X_train_clusterID], axis=1)



#-----------------------------------------------
# Finde die wichtigsten Anlagen für jedes Cluster
#-----------------------------------------------

# Die wichtigsten Anlagen sind diejenigen, die mit einem Anteil von
# relevance_level unter allen Gebäuden des Clusters vorhanden sind.
relevance_level = 0.9

# Speichere für jedes Cluster die wichtigen Anlagen in einem Dictionary.
var_dict = {}
for i in range(n_clusters):
    current_cluster = X_train_clustered[X_train_clustered["ClusterID"] == i
                                       ].drop(columns="ClusterID")
    var_dict[f"equipments{i}"] = current_cluster.sum(
        axis=0).sort_values(ascending=False)
    var_dict[f"equipments{i}"] = var_dict[f"equipments{i}"][
        var_dict[f"equipments{i}"] >=
        relevance_level * len(current_cluster)].index



#-----------------------------------------------
# Überprüfe das Test Set auf fehlende Anlagen
#-----------------------------------------------

# Finde die richtigen Cluster für die Gebäude
X_test_clusterID = pd.DataFrame(
                        kmeans.predict(X_test),
                        columns=["ClusterID"], index=X_test.index)
X_test_clustered = pd.concat([X_test, X_test_clusterID], axis=1)


# Entkommentieren, um Test Set auf fehlende Anlagen zu checken
# =============================================================================
# print("\n TEST SET:")
# for building in X_test_clustered.index:
#     current_cluster = X_test_clustered.loc[building, "ClusterID"]
#     curr_missing = []
#     for equip in var_dict[f"equipments{current_cluster}"]:
#         if X_test_clustered.loc[building, equip] == 0:
#             curr_missing.append(equip)
#     if curr_missing:
#         print(f"\n    Verdächtig: Das Gebäude {building} sollte typischerweise noch folgende Anlagen enthalten: ")
#         for anlage in curr_missing:
#             print(f"       {anlage}")
# =============================================================================



#-----------------------------------------------
# Kundendatei auf fehlende Anlagen überprüfen
#-----------------------------------------------

# Importiere die in zuordnung.py erstellte Anlagendatei
# Die Anlagen-IDs aus dem EP-Katalog sind also schon automatisch zugeordnet.

kunde = pd.read_csv("Kundendatei Anlagenzuordnung.csv")
kunde = kunde[["WirtEinh", "Pred ID"]]
kunde = kunde.drop_duplicates()

# Transformiere Kundendatei, ergänze fehlende Anlagennummern
kunde_pivot = pd.crosstab(kunde["WirtEinh"], kunde["Pred ID"])
for column in df_pivot.columns:
    if column not in kunde_pivot.columns:
        kunde_pivot[column] = 0

# Berechne Cluster der Kunden-Gebäude
kunde_cluster_id = pd.DataFrame(
                    kmeans.fit_predict(kunde_pivot), 
                    columns=["ClusterID"], index=kunde_pivot.index)
kunde_clustered = pd.concat([kunde_pivot, kunde_cluster_id], axis=1)

# Überprüfe auf fehlende Anlagen für jedes Gebäude
print("\n KUNDENDATEI:")
for building in kunde_clustered.index:
    current_cluster = kunde_clustered.loc[building, "ClusterID"]
    curr_missing = []
    for equip in var_dict[f"equipments{current_cluster}"]:
        if kunde_clustered.loc[building, equip] == 0:
            curr_missing.append(equip)
    if curr_missing:
        print(f"    Verdächtig: Das Gebäude {building} sollte typischerweise noch {len(curr_missing)} weitere Anlagen enthalten.")
