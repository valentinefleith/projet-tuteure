import os
import ollama
import pandas as pd
import glob
from evaluation import Metrics, evaluate_annotation, pretty_print
from tqdm import tqdm
from sklearn.utils import resample

MODEL = "deepseek-r1:1.5b"
CSV_PATH = "corpus/corpus_phrases/test_prompt"
ANNOTATIONS_PATH = f"annotations/annotations_llm/{MODEL}"
RESULTS_PATH = f"results/classification/{MODEL}"


def annotate_with_ollama(sentences: pd.DataFrame) -> list:
    results = []
    for sentence in tqdm(sentences, desc="Annotation en cours"):
        response = ollama.chat(
            model="urbaniste", messages=[{"role": "user", "content": sentence}]
        )
        annotation = response["message"]["content"].strip()

        if MODEL.startswith("deepseek"):
            print(f"🟡 DEBUG - Réponse brute du modèle : {annotation}")
        try:
            annotation = (
                annotation[-1] if MODEL.startswith("deepseek") else annotation[0]
            )
            results.append(int(annotation))
        except (ValueError, IndexError):
            print(f"⚠️ Erreur : Réponse inattendue -> {annotation}")
            results.append(-1)

    return results


def get_downsampled(df) -> pd.DataFrame:
    class_counts = df["dynamic"].value_counts()
    biggest_class = class_counts.idxmax()
    # we separate the majority class from the rest of the samples
    biggest_class_df = df.query(f"`dynamic` == {biggest_class}")
    df_without_biggest = df.query(f"`dynamic` != {biggest_class}")
    resampled_class = resample(
        biggest_class_df,
        replace=False,
        n_samples=int(class_counts.median()),  # reduce n_sample to median
        random_state=42,
    )
    # and then concatenate them after reducing the size
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

    print("🟡 INFO : Suppression des annotations invalides")
    nb_of_invalid = len(downsampled_df["annotation"] == -1)
    print(f"{nb_of_invalid} lignes sont invalides et donc supprimees.")
    downsampled_df = downsampled_df[downsampled_df["annotation"] != -1]

    if save:
        os.makedirs(ANNOTATIONS_PATH, exist_ok=True)
        downsampled_df.to_csv(
            f"{ANNOTATIONS_PATH}/{os.path.basename(csv_file)}", sep="|", index=False
        )

    return downsampled_df


def save_results(metrics: Metrics, conf_matrix, filename):
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
    all_csv_files = glob.glob(CSV_PATH + "/*.csv")
    for csv_file in all_csv_files:
        filename = csv_file.split("/")[-1]
        print(f"Currently loading {filename}...")
        if os.path.isfile(f"{RESULTS_PATH}/{filename}"):
            continue
        annotated_df = get_annotated_df(csv_file)
        evaluation, conf_matrix = evaluate_annotation(annotated_df)
        pretty_print(evaluation, conf_matrix, filename)
        save_results(evaluation, conf_matrix, filename)


if __name__ == "__main__":
    main()
