import os
import matplotlib.pyplot as plt
import pandas as pd

def save_report(results_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, "results_grid.csv"), index=False)
    pivot = results_df.pivot(index="Model", columns="Embedding", values="Accuracy")
    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values, cmap="YlGnBu", vmin=0.4, vmax=1.0)
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="black")
    plt.title("Accuracy: Embedding x Model")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "results_heatmap.png"), dpi=150)
    plt.close()
    best_row = results_df.loc[results_df["Accuracy"].idxmax()]
    best_model = best_row["Model"]
    best_embedding = best_row["Embedding"]
    best_accuracy = best_row["Accuracy"]
    with open(os.path.join(out_dir, "results_report.txt"), "w") as f:
        f.write("Restaurant Review Sentiment Analysis - Embedding x Model Comparison\n")
        f.write("=" * 70 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nBest combination:\n")
        f.write(f"  Model     : {best_model}\n")
        f.write(f"  Embedding : {best_embedding}\n")
        f.write(f"  Accuracy  : {best_accuracy:.3f}\n")
    print("\nSaved: results_grid.csv, results_heatmap.png, results_report.txt")