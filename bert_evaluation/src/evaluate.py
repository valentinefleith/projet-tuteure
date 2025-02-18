"""
Module d'évaluation des annotations automatiques contre les annotations manuelles.

Ce script compare les annotations extraites automatiquement avec celles annotées manuellement,
en utilisant la similarité de Jaccard et un seuil dynamique.

📌 Fonctionnalités :
- Calcul de la précision, du rappel et de la F-mesure pour différentes dimensions.
- Comparaison des annotations auto et manuelles.
- Sauvegarde des résultats individuels et d'un résumé global.

📥 Entrée :
- Annotations automatiques : dossiers "annotations/annotations_auto/"
- Annotations manuelles : dossiers "annotations/annotations_manuelles/"

📤 Sortie :
- Résultats par fichier annoté dans "resultats/{nom_fichier}_results.csv"
- Résumé global dans "resultats/evaluation_summary.csv"
"""

import pandas as pd
import glob
import os

# Dimensions à traiter
DIMENSIONS = ["dyn", "act", "perc", "tps", "doc", "lieu_loc", "lieu_obj"]
ANNOTATIONS_AUTO = "annotations/annotations_auto"
ANNOTATIONS_MANUELLES = "annotations/annotations_manuelles"
RESULTS_DIR = "resultats"
SEUIL_SIMILARITE = 0.5

# Création du dossier résultats s'il n'existe pas
os.makedirs(RESULTS_DIR, exist_ok=True)


def jaccard_similarity(text1, text2):
    """
    Calcule la similarité de Jaccard entre deux segments.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Paramètres :
    - text1 (str) : Premier segment de texte.
    - text2 (str) : Deuxième segment de texte.

    Retourne :
    - float : Valeur entre 0 et 1 représentant la similarité.
    """
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def dynamic_jaccard_threshold(text1, text2):
    """
    Ajuste dynamiquement le seuil de Jaccard en fonction de la longueur moyenne des textes.

    Paramètres :
    - text1 (str) : Premier segment de texte.
    - text2 (str) : Deuxième segment de texte.

    Retourne :
    - float : Seuil de similarité ajusté (entre 0.5 et 0.8).
    """
    len_text1 = len(text1.split())
    len_text2 = len(text2.split())
    avg_length = (len_text1 + len_text2) / 2

    if avg_length < 5:
        return 0.8
    elif avg_length < 10:
        return 0.6
    else:
        return 0.5


def normalize_text(text):
    """
    Normalise un texte en supprimant les espaces et en le mettant en minuscules.

    Paramètres :
    - text (str) : Texte à normaliser.

    Retourne :
    - str : Texte normalisé.
    """
    if isinstance(text, str):
        return text.strip().lower()
    return ""


def normalize_and_check_empty(segments_bert, segments_reels, dim):
    """
    Normalise les segments et vérifie si la colonne d'annotations automatiques est vide.

    Paramètres :
    - segments_bert (pd.DataFrame) : Annotations automatiques.
    - segments_reels (pd.DataFrame) : Annotations manuelles.
    - dim (str) : Dimension de l'annotation.

    Retourne :
    - pd.DataFrame : Segments normalisés (ou None si vide).
    - pd.DataFrame : Segments manuels normalisés.
    - bool : True si la colonne est vide, False sinon.
    """
    segments_reels = segments_reels.copy()
    segments_bert = segments_bert.copy()

    if segments_bert[f"balise_{dim}"].isna().all():
        print(f"⚠️  Aucune annotation trouvée pour '{dim}', métriques mises à zéro.")
        return None, None, True

    segments_reels["Segment_Annoté"] = (
        segments_reels["Segment_Annoté"].astype(str).apply(normalize_text)
    )
    segments_bert[f"balise_{dim}"] = (
        segments_bert[f"balise_{dim}"].astype(str).apply(normalize_text)
    )

    return segments_bert, segments_reels, False


def compute_confusion_matrix(segments_bert, segments_reels, dim):
    """
    Calcule les valeurs TP, FP et FN pour une dimension donnée.

    Paramètres :
    - segments_bert (pd.DataFrame) : Annotations automatiques.
    - segments_reels (pd.DataFrame) : Annotations manuelles.
    - dim (str) : Dimension de l'annotation.

    Retourne :
    - tuple : (TP, FP, FN)
    """
    segments_reels_list = segments_reels["Segment_Annoté"].tolist()

    matching_rows = segments_bert[
        segments_bert[f"balise_{dim}"].apply(
            lambda x: any(
                jaccard_similarity(x, s) >= dynamic_jaccard_threshold(x, s)
                for s in segments_reels_list
            )
        )
    ]

    TP = len(matching_rows)
    FP = len(segments_bert) - TP
    FN = len(segments_reels) - TP

    return TP, FP, FN


def compute_precision_recall_f1(TP, FP, FN):
    """
    Calcule la précision, le rappel et la F-mesure.

    Paramètres :
    - TP (int) : Vrais positifs.
    - FP (int) : Faux positifs.
    - FN (int) : Faux négatifs.

    Retourne :
    - tuple : (précision, rappel, F-mesure)
    """
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1_score


def evaluate_matching(segments_bert, segments_reels, dim):
    """
    Évalue la performance de l'annotation automatique pour une dimension donnée.

    Paramètres :
    - segments_bert (pd.DataFrame) : Annotations automatiques.
    - segments_reels (pd.DataFrame) : Annotations manuelles.
    - dim (str) : Dimension de l'annotation.

    Retourne :
    - tuple : (précision, rappel, F-mesure)
    """
    segments_bert, segments_reels, is_empty = normalize_and_check_empty(
        segments_bert, segments_reels, dim
    )

    if is_empty:
        return 0, 0, 0

    TP, FP, FN = compute_confusion_matrix(segments_bert, segments_reels, dim)
    return compute_precision_recall_f1(TP, FP, FN)


def compute_evaluation(filename):
    """
    Évalue les annotations automatiques pour un fichier donné.

    Cette fonction charge les annotations automatiques et manuelles,
    puis calcule la précision, le rappel et la F-mesure pour chaque dimension.

    Paramètres :
    - filename (str) : Chemin du fichier d'annotations automatiques.

    Retourne :
    - list[dict] : Liste des résultats sous forme de dictionnaires.
                   Chaque dictionnaire contient le fichier traité,
                   la dimension, la précision, le rappel et la F-mesure.
    """
    name_in_table = f"{os.path.basename(filename).split('.')[0]}.xml"
    file_results = []

    for dim in DIMENSIONS:
        print(f"📌 Traitement du fichier {filename} - Dimension: {dim}")

        # Charger les annotations manuelles
        if dim.startswith("lieu_"):
            manual_path = f"{ANNOTATIONS_MANUELLES}/all_lieu.tsv"
        else:
            manual_path = f"{ANNOTATIONS_MANUELLES}/all_{dim}.tsv"

        if not os.path.exists(manual_path):
            print(
                f"⚠️  Fichier d'annotations manuelles manquant : {manual_path}. Skipping..."
            )
            continue  # Passer à la dimension suivante

        ground_truth_df = pd.read_csv(manual_path, sep="\t").query(
            "`Nom_fichier` == @name_in_table"
        )

        # Charger les annotations automatiques
        try:
            annotation_auto = pd.read_csv(filename)
        except pd.errors.ParserError:
            annotation_auto = pd.read_csv(filename, sep="|")

        # Sélectionner les colonnes pertinentes
        if dim.startswith("lieu_"):
            segments_reels = ground_truth_df[
                ground_truth_df["Type"] == dim.split("_")[-1]
            ][["Segment_Annoté"]]
        else:
            segments_reels = ground_truth_df[["Segment_Annoté"]]
        segments_bert = annotation_auto[[f"balise_{dim}", "phrase", "phrase_id"]]

        # Calcul des métriques
        precision, recall, f1_score = evaluate_matching(
            segments_bert, segments_reels, dim
        )

        # Stocker les résultats pour ce fichier et cette dimension
        file_results.append(
            {
                "Nom_fichier": name_in_table,
                "Dimension": dim,
                "Précision": precision,
                "Rappel": recall,
                "F-mesure": f1_score,
            }
        )

    return file_results


def main():
    """
    Fonction principale qui traite tous les fichiers d'annotations automatiques.

    1. Récupère la liste des fichiers à traiter dans le dossier "annotations_auto".
    2. Calcule les métriques d'évaluation pour chaque fichier.
    3. Enregistre les résultats individuels et crée un fichier récapitulatif global.

    Sorties :
    - Résultats individuels : "resultats/{nom_fichier}_results.csv"
    - Résumé global : "resultats/evaluation_summary.csv"
    """
    all_files = glob.glob(f"{ANNOTATIONS_AUTO}/*.tsv")
    global_results = []

    if not all_files:
        print(
            "⚠️  Aucun fichier trouvé dans 'annotations_auto/'. Vérifiez votre dossier."
        )
        return

    for file in all_files:
        file_results = compute_evaluation(file)
        global_results.extend(file_results)

        # Sauvegarde des résultats individuels par fichier
        if file_results:
            file_results_df = pd.DataFrame(file_results)
            file_results_path = os.path.join(
                RESULTS_DIR, f"{os.path.basename(file)}_results.csv"
            )
            file_results_df.to_csv(file_results_path, index=False, sep="\t")

            print(f"✅ Résultats enregistrés dans {file_results_path}")

    if global_results:
        # Création du fichier global avec les moyennes
        global_df = pd.DataFrame(global_results)

        # Calcul des moyennes par dimension
        summary_df = (
            global_df.groupby("Dimension")[["Précision", "Rappel", "F-mesure"]]
            .mean()
            .reset_index()
        )

        summary_file = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
        summary_df.to_csv(summary_file, index=False, sep="\t")

        print(f"\n📊 Résumé global enregistré dans {summary_file}")
    else:
        print(
            "\n⚠️  Aucun résultat calculé. Vérifiez que les fichiers sont bien formatés et présents."
        )


if __name__ == "__main__":
    main()
