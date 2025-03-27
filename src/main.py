import os
import ollama
import pandas as pd
import glob
from evaluation import Metrics, evaluate_annotation, pretty_print
from tqdm import tqdm
from sklearn.utils import resample

MODEL = "deepseek-r1:1.5b"
CSV_PATH = "corpus/corpus_phrases"
ANNOTATIONS_PATH = f"annotations/annotations_llm/{MODEL}"
RESULTS_PATH = f"results/classification/{MODEL}"

def annotate_with_ollama(sentences: list, df: pd.DataFrame, save_path: str) -> list:
    """
    Envoie les phrases à Ollama pour annotation et met à jour le DataFrame ligne par ligne.
    Sauvegarde après chaque annotation pour éviter de tout recommencer en cas d'arrêt.
    """
    results = []
    non_annotated_indices = df[df["annotation"] == -1].index.tolist()

    for i, (sentence, index) in enumerate(tqdm(zip(sentences, non_annotated_indices), 
                                               total=len(sentences), 
                                               desc="Annotation en cours", 
                                               leave=True)):
        response = ollama.chat(
            model="urbaniste", messages=[{"role": "user", "content": sentence}]
        )
        annotation = response["message"]["content"].strip()

        if MODEL.startswith("deepseek"):
            annotation = "".join(filter(str.isdigit, annotation))[:1]

        try:
            annotation_int = int(annotation)
            if annotation_int in [0, 1]:
                results.append(annotation_int)
            else:
                raise ValueError("Valeur inattendue")
        except (ValueError, IndexError):
            print(f"⚠️ Erreur : Réponse inattendue -> {annotation}")
            results.append(-1)

        # 🔥 Vérification stricte : mise à jour ligne par ligne
        df.at[index, "annotation"] = results[-1]
        df.to_csv(save_path, sep="|", index=False)

    return results


def get_annotated_df(csv_file: str, save=True) -> pd.DataFrame:
    """
    Charge un fichier CSV et annote les phrases.
    """

    save_path = f"{ANNOTATIONS_PATH}/{os.path.basename(csv_file)}"

    if MODEL.startswith("deepseek") and os.path.exists(save_path):
        print(f"🟡 INFO : Chargement des annotations incomplètes depuis {save_path}")
        df = pd.read_csv(save_path, sep="|")

        if "annotation" not in df.columns:
            df["annotation"] = -1

        non_annotated_mask = df["annotation"] == -1
        sentences_to_annotate = df.loc[non_annotated_mask, "sentence"].tolist()
    else:
        print(f"🔵 INFO : Aucune annotation existante, démarrage depuis zéro")
        df = pd.read_csv(csv_file, sep="|")
        df["annotation"] = -1  # Ajoute la colonne annotation initialisée à -1
        sentences_to_annotate = df["sentence"].tolist()

    if len(sentences_to_annotate) == 0:
        print(f"✅ INFO : Toutes les phrases de {csv_file} sont déjà annotées !")
        return df

    annotations = annotate_with_ollama(sentences_to_annotate, df, save_path)

    if len(annotations) != len(sentences_to_annotate):
        print(f"❌ Erreur : Nombre d'annotations ({len(annotations)}) ≠ Phrases à annoter ({len(sentences_to_annotate)})")
        print("🚨 ANNULATION de la mise à jour des annotations pour éviter corruption.")
        return df

    df.loc[df["annotation"] == -1, "annotation"] = annotations

    if MODEL.startswith("deepseek"):
        print("🟡 INFO : Suppression des annotations invalides pour DeepSeek")
        df = df[df["annotation"] != -1]

    if save:
        os.makedirs(ANNOTATIONS_PATH, exist_ok=True)
        df.to_csv(save_path, sep="|", index=False)
        print(f"💾 INFO : Fichier d'annotations mis à jour -> {save_path}")

    return df


def save_results(metrics: Metrics, conf_matrix, filename):
    """
    Enregistre les résultats de classification et la matrice de confusion.
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Score": [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1],
        }
    )
    metrics_df.to_csv(f"{RESULTS_PATH}/{filename}", index=False)
    
    conf_matrix_df = pd.DataFrame(
        conf_matrix, columns=["Predicted False", "Predicted Dynamic"]
    )
    conf_matrix_df.index = ["Actual False", "Actual Dynamic"]
    conf_matrix_df.to_csv(f"{RESULTS_PATH}/{filename}_confusion_matrix")


def main():
    """
    Exécute l'annotation et l'évaluation sur tous les fichiers CSV du corpus.
    """
    all_csv_files = glob.glob(CSV_PATH + "/*.csv")
    for csv_file in all_csv_files:
        filename = os.path.basename(csv_file)
        print(f"\n📌 Traitement du fichier : {filename}\n")
        
        annotated_df = get_annotated_df(csv_file)
        evaluation, conf_matrix = evaluate_annotation(annotated_df)
        
        pretty_print(evaluation, conf_matrix, filename)
        save_results(evaluation, conf_matrix, filename)


if __name__ == "__main__":
    main()
