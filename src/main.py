import os
import ollama
import pandas as pd
import glob
from evaluation import Metrics, evaluate_annotation, pretty_print

MODEL = "mistral"
CSV_PATH = "corpus/corpus_phrases"
ANNOTATIONS_PATH = f"annotations/annotations_llm/{MODEL}"
RESULTS_PATH = f"results/classification/{MODEL}"


def annotate_with_ollama(sentences: pd.DataFrame) -> list:
    results = []
    for sentence in sentences:
        response = ollama.chat(
            model="urbaniste", messages=[{"role": "user", "content": sentence}]
        )
        annotation = response["message"]["content"]
        results.append(int(annotation))
    return results


def get_annotated_df(csv_file: str, save=True) -> pd.DataFrame:
    df = pd.read_csv(csv_file, sep="|")
    sentences = df["sentence"].tolist()
    annotations = annotate_with_ollama(sentences)
    df["annotation"] = annotations
    if save:
        os.makedirs(ANNOTATIONS_PATH, exist_ok=True)
        df.to_csv(f"{ANNOTATIONS_PATH}/{csv_file.split('/')[-1]}", sep="|", index=False)
    return df


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
        annotated_df = get_annotated_df(csv_file)
        evaluation, conf_matrix = evaluate_annotation(annotated_df)
        pretty_print(evaluation, conf_matrix, filename)
        save_results(evaluation, conf_matrix, filename)


if __name__ == "__main__":
    main()
