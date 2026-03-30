import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, roc_auc_score

from opensoundscape import BoxedAnnotations
import bioacoustics_model_zoo as bmz
import torch


# -----------------
# Config
# -----------------
run_name = "perch2_shallow_classifier"

metadata_path = Path("/mimer/NOBACKUP/groups/rewilding_madagascar/data/metadata_mimer.csv")
out_dir = Path(f"/mimer/NOBACKUP/groups/rewilding_madagascar/models/{run_name}")

state_dict_path = out_dir / "classifier_state_dict.pt"   # <-- uses your existing weights file

class_list = [
    'Hypsipetes_madagascariensis','Copsychus_albospecularis','Coracopsis_nigra','Dicrurus_forficatus',
    'Coua_caerulea','Zosterops_maderaspatanus','Eurystomus_glaucurus','Agapornis_canus','Saxicola_torquatus',
    'Cyanolanius_madagascarinus','Leptopterus_chabert','Nesoenas_picturatus','Coua_reynaudii',
    'Ceblepyris_cinereus','Neodrepanis_coruscans','Philepitta_castanea','Eulemur_sp','Coua_cristata',
    'Treron_australis'
]


# -----------------
# Load metadata + build val_labels (same as training)
# -----------------
metadata = pd.read_csv(metadata_path)
val_metadata = metadata[metadata["Split"] == "validation"].reset_index(drop=True)

val_annotations = BoxedAnnotations.from_raven_files(
    val_metadata["Raven_path"], "species", val_metadata["SoundFile_path"]
)

conversion_table = pd.DataFrame(
    {"original": ["Eulemur_albifrons", "Eulemur_fulvus"],
     "new": ["Eulemur_sp", "Eulemur_sp"]}
)

val_annotations_corrected = val_annotations.convert_labels(conversion_table)
val_annotations_corrected.audio_files = val_annotations_corrected.df['audio_file'].values  # workaround #872

val_labels = val_annotations_corrected.clip_labels(
    clip_duration=5,
    clip_overlap=0,
    min_label_overlap=0.25,
    class_subset=class_list
)


# -----------------
# Rebuild model + load weights (NO pickle)
# -----------------
perch2_model = bmz.Perch2()
perch2_model.initialize_custom_classifier(class_list, hidden_layer_sizes=())

sd = torch.load(state_dict_path, weights_only=False)  # safe: file you created
perch2_model.network.load_state_dict(sd)
perch2_model.network.eval()

print(f"Loaded classifier weights from: {state_dict_path}")


# -----------------
# Evaluation (your style + sigmoid fix)
# -----------------
def evaluate_on_val(model, val_labels, class_list, out_dir, filename, batch_size=64, bins=20):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_dir = out_dir / "histograms_prob"
    hist_semilog_dir = hist_dir / "semilog"
    results_dir = out_dir / "results_prob"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_semilog_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    preds = model.predict(val_labels, batch_size=batch_size)
    print("[eval] preds columns (first 10):", list(preds.columns)[:10])

    preds = preds[class_list].copy()

    # logits -> probabilities
    preds = preds.clip(-50, 50)
    preds = 1 / (1 + np.exp(-preds))

    preds_path = results_dir / f"{filename}_val_preds_prob.csv"
    preds.to_csv(preds_path)
    print(f"[eval] Wrote prob predictions: {preds_path}")

    scores_valid_df = val_labels.join(preds, rsuffix="pred")
    plt.rcParams["figure.figsize"] = [15, 5]

    for species in class_list:
        pred_col = species + "pred"
        if pred_col not in scores_valid_df.columns:
            print(f"[eval] WARNING: missing prediction column {pred_col}, skipping.")
            continue

        df_pos = scores_valid_df[scores_valid_df[species] == True]
        df_not = scores_valid_df[scores_valid_df[species] == False]

        plt.hist(df_not[pred_col], bins=bins, alpha=0.5, label="negatives")
        plt.hist(df_pos[pred_col], bins=bins, alpha=0.5, label="positives")
        plt.legend()
        plt.xlabel("Predicted probability")
        plt.ylabel("Frequency")
        plt.savefig(hist_dir / f"{filename}_{species}.png", dpi=150, bbox_inches="tight")
        plt.clf()

        plt.hist(df_not[pred_col], bins=bins, alpha=0.5, label="negatives")
        plt.hist(df_pos[pred_col], bins=bins, alpha=0.5, label="positives")
        plt.legend()
        plt.xlabel("Predicted probability")
        plt.ylabel("Frequency")
        plt.semilogy()
        plt.savefig(hist_semilog_dir / f"{filename}_{species}.png", dpi=150, bbox_inches="tight")
        plt.clf()

    print(f"[eval] Wrote histograms: {hist_dir}")

    rows = []
    for species in class_list:
        y_true = val_labels[species].astype(int).values
        y_score = preds[species].astype(float).values
        ap = float(average_precision_score(y_true, y_score))
        auc = np.nan if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_score))
        rows.append({"species": species, "avg_precision_score": ap, "auroc_score": auc})

    metrics_df = pd.DataFrame(rows)
    metrics_path = results_dir / f"{filename}_metrics_by_species_prob.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[eval] Wrote metrics: {metrics_path}")

    return preds, metrics_df


preds, metrics = evaluate_on_val(
    model=perch2_model,
    val_labels=val_labels,
    class_list=class_list,
    out_dir=out_dir,
    filename=run_name,
    batch_size=64,
    bins=20,
)