import os
import ollama
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.utils import resample
from evaluation import Metrics, evaluate_annotation, pretty_print

# Utilisation de DeepSeek-r1:1.5b
MODEL = "deepseek-r1:1.5b"

# Définition des chemins d'entrée/sortie
CSV_PATH = "corpus/corpus_phrases"
ANNOTATIONS_PATH = f"annotations/annotations_llm/{MODEL}"
RESULTS_PATH = f"results/classification/{MODEL}"

def annotate_with_ollama(sentences: pd.DataFrame) -> list:
    """
    Envoie les phrases au modèle DeepSeek et récupère les annotations.
    """
    results = []
    for sentence in tqdm(sentences, desc="Annotation en cours"):
        response = ollama.chat(
            model="urbaniste", messages=[{"role": "user", "content": sentence}]
        )
        annotation = response["message"]["content"].strip()

        print(f"🟡 DEBUG - Réponse brute du modèle : {annotation}")  # Afficher la réponse brute

        # Ne garder que les chiffres (0 ou 1) au début de la réponse
        annotation = ''.join(filter(str.isdigit, annotation))[:1]

        if annotation in ["0", "1"]:
            results.append(int(annotation))
        else:
            print(f"⚠️ Erreur : Réponse inattendue -> {annotation}")  # Debug
            results.append(-1)  # Valeur par défaut pour éviter le crash

    return results

def get_downsampled(df) -> pd.DataFrame:
    """
    Réduit la classe majoritaire pour équilibrer les données.
    """
    class_counts = df["dynamic"].value_counts()
    biggest_class = class_counts.idxmax()

    # Séparation des classes
    biggest_class_df = df[df["dynamic"] == biggest_class]
    df_without_biggest = df[df["dynamic"] != biggest_class]

    # Réduction de la classe majoritaire au niveau médian
    resampled_class = resample(
        biggest_class_df,
        replace=False,
        n_samples=int(class_counts.median()), 
        random_state=42,
    )

    # Concatenation et mélange des données équilibrées
    return (
        pd.concat([df_without_biggest, resampled_class])
        .sample(frac=1)
        .reset_index(drop=True)
    )

def get_annotated_df(csv_file: str, save=True) -> pd.DataFrame:
    """
    Charge un fichier CSV, applique un downsampling et annote les phrases.
    """
    df = pd.read_csv(csv_file, sep="|")
    downsampled_df = get_downsampled(df)
    sentences = downsampled_df["sentence"].tolist()
    annotations = annotate_with_ollama(sentences)
    downsampled_df["annotation"] = annotations

    # Suppression des annotations invalides (-1)
    downsampled_df = downsampled_df[downsampled_df["annotation"] != -1]

    if save:
        os.makedirs(ANNOTATIONS_PATH, exist_ok=True)
        downsampled_df.to_csv(f"{ANNOTATIONS_PATH}/{os.path.basename(csv_file)}", sep="|", index=False)

    return downsampled_df

def save_results(metrics: Metrics, conf_matrix, filename):
    """
    Enregistre les métriques et la matrice de confusion.
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
