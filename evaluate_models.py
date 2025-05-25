from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, \
    confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(model, X_test, y_test, name="", label_names=None, save_results=True):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'rmse': rmse,
        'mae': mae
    }

    print(f"\n==== {name} ====")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    if save_results:
        os.makedirs("results", exist_ok=True)
        with open(f"results/{name}_report.txt", "w") as f:
            f.write(classification_report(y_test, y_pred))
            for metric, value in metrics.items():
                f.write(f"\n{metric.capitalize()}: {value:.4f}")

        if label_names is not None:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
            disp.plot(cmap='Blues', xticks_rotation='vertical')
            plt.title(f"{name} Confusion Matrix")
            plt.tight_layout()
            plt.savefig(f"results/conf_matrix_{name}.png")
            plt.close()

        # Call the plot function to visualize metrics
        plot_evaluation_results(metrics, name)

    return metrics

def plot_evaluation_results(metrics, model_name, results_dir="results"):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = [["Metric", "Value"]] + [[key.capitalize(), f"{value:.4f}"] for key, value in metrics.items()]

    # Create the table
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save the table as an image
    plot_path = os.path.join(results_dir, f"{model_name}_evaluation_table.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Evaluation metrics table saved to {plot_path}")