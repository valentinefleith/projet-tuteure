from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from collections import namedtuple

Metrics = namedtuple("Metrics", ["accuracy", "precision", "recall", "f1"])


def evaluate_annotation(annotated_df):
    y_true = annotated_df["dynamic"].tolist()
    y_pred = annotated_df["annotation"].tolist()
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    metrics = Metrics(accuracy, precision, recall, f1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return metrics, conf_matrix


def pretty_print(metrics: Metrics, conf_matrix, filename) -> None:
    console = Console()
    # METRICS
    console.print(f"\n[bold cyan]Results for {filename}[/bold cyan]\n")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="bold yellow")

    table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    table.add_row("Precision", f"{metrics.precision:.4f}")
    table.add_row("Recall", f"{metrics.recall:.4f}")
    table.add_row("F1-score", f"{metrics.f1:.4f}")

    console.print(table)

    # CONFUSION MATRIX
    console.print("\n[bold cyan]Confusion Matrix[/bold cyan]")
    conf_table = Table(header_style="bold red")
    conf_table.add_column("")
    conf_table.add_column("Predicted False", justify="center", style="bold green")
    conf_table.add_column("Predicted Dynamic", justify="center", style="bold green")

    conf_table.add_row("Actual False", str(conf_matrix[0][0]), str(conf_matrix[0][1]))
    conf_table.add_row("Actual Dynamic", str(conf_matrix[1][0]), str(conf_matrix[1][1]))

    console.print(conf_table)
