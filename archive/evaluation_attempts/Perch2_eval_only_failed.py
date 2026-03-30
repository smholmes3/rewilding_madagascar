# -----------------
# Evaluation
# -----------------

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless batch jobs
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

def evaluate_on_val(
    model,
    val_labels: pd.DataFrame,
    class_list: list[str],
    out_dir: Path,
    filename: str,
    batch_size: int = 64,
    bins: int = 20,
):
    """
    model: bmz.Perch2() object (already trained / has custom classifier)
    val_labels: DataFrame of clip labels (index can be multiindex; columns include class_list + metadata)
    class_list: list of class names (strings)
    out_dir: directory to save outputs
    filename: base name for outputs
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subfolders
    hist_dir = out_dir / "histograms"
    hist_semilog_dir = hist_dir / "semilog"
    results_dir = out_dir / "results"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_semilog_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Predict
    # IMPORTANT: method signature may vary slightly by bmz version.
    # In your Perch notebook you used: model.predict(val_labels, batch_size=64)
    preds = model.predict(val_labels, batch_size=batch_size)
    print("[eval] preds columns (first 10):", list(preds.columns)[:10])

    # Keep only the class columns (and in consistent order)
    preds = preds[class_list].copy()
    # Convert logits -> probabilities for plotting and metrics
    preds = preds.clip(-50, 50)              # avoid overflow
    preds = 1 / (1 + np.exp(-preds))         # sigmoid

    # Save raw predictions
    preds_path = results_dir / f"{filename}_val_preds.csv"
    preds.to_csv(preds_path)
    print(f"[eval] Wrote predictions: {preds_path}")

    # Join labels + preds for plotting (add suffix 'pred' to prediction columns)
    # This matches your old approach where you referenced '<species>pred'
    scores_valid_df = val_labels.join(preds, rsuffix="pred")

    # ---- 2) Histograms (linear + semilog)
    plt.rcParams["figure.figsize"] = [15, 5]

    for species in class_list:
        pred_col = species

        # Some safety: skip if prediction column isn't present
        if pred_col not in scores_valid_df.columns:
            print(f"[eval] WARNING: missing prediction column {pred_col}, skipping histograms.")
            continue

        df_pos = scores_valid_df[scores_valid_df[species] == True]
        df_not = scores_valid_df[scores_valid_df[species] == False]

        # Linear histogram
        plt.hist(df_not[pred_col], bins=bins, alpha=0.5, label="negatives")
        plt.hist(df_pos[pred_col], bins=bins, alpha=0.5, label="positives")
        plt.legend()
        plt.xlabel("Predicted probability")
        plt.ylabel("Frequency")
        out_png = hist_dir / f"{filename}_{species}.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.clf()

        # Semilog histogram (y-axis)
        plt.hist(df_not[pred_col], bins=bins, alpha=0.5, label="negatives")
        plt.hist(df_pos[pred_col], bins=bins, alpha=0.5, label="positives")
        plt.legend()
        plt.xlabel("Predicted probability")
        plt.ylabel("Frequency")
        plt.semilogy()
        out_png = hist_semilog_dir / f"{filename}_{species}.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.clf()

    print(f"[eval] Wrote histograms: {hist_dir}")

    # ---- 3) Metrics by species (AP + AUROC)
    rows = []
    for species in class_list:
        y_true = val_labels[species].astype(int).values
        y_score = preds[species].astype(float).values

        ap = float(average_precision_score(y_true, y_score))

        # AUROC is undefined if only one class present in y_true
        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(y_true, y_score))

        rows.append({"species": species, "avg_precision_score": ap, "auroc_score": auc})

    metrics_df = pd.DataFrame(rows)
    metrics_path = results_dir / f"{filename}_metrics_by_species.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[eval] Wrote metrics: {metrics_path}")

    return preds, metrics_df

preds, metrics = evaluate_on_val(
    model=perch2_model,
    val_labels=val_labels,
    class_list=class_list,
    out_dir=out_dir,        # directory for histograms/results
    filename=run_name,      # base name for files
    batch_size=64,
)

