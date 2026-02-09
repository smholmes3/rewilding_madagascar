#Import all the libraries we need
import pandas as pd
from pathlib import Path
# opensoundscape transfer learning tools
from opensoundscape import BoxedAnnotations
import bioacoustics_model_zoo as bmz

# -----------------
# Config
# -----------------
filename = "perch2_shallow_classifier"

# Prefer absolute paths on HPC
metadata_path = Path("/mimer/NOBACKUP/groups/rewilding_madagascar/data/metadata_mimer.csv")
save_path = Path(f"/mimer/NOBACKUP/groups/rewilding_madagascar/models/{filename}")

# -----------------
# Load metadata
# -----------------
metadata = pd.read_csv(metadata_path)

#Step 1: split the metadata into train, validation, and test sets
val_metadata = metadata[metadata["Split"] == "validation"].reset_index(drop=True)
train_metadata = metadata[metadata["Split"] == "train"].reset_index(drop=True)
test_metadata = metadata[metadata["Split"] == "test"].reset_index(drop=True)

# Step 2: load the annotations into OpenSoundscape
val_annotations = BoxedAnnotations.from_raven_files(
    val_metadata["Raven_path"], "species", val_metadata["SoundFile_path"]
)
train_annotations = BoxedAnnotations.from_raven_files(
    train_metadata["Raven_path"], "species", train_metadata["SoundFile_path"]
)
test_annotations = BoxedAnnotations.from_raven_files(
    test_metadata["Raven_path"], "species", test_metadata["SoundFile_path"]
)

# Step 3: create a conversion table to map the original species names to the new ones we want to use for training
conversion_table = pd.DataFrame(
    {"original": ["Eulemur_albifrons", "Eulemur_fulvus"],
     "new": ["Eulemur_sp", "Eulemur_sp"]}
)

#Step 4: correct annotations in each of the splits
val_annotations_corrected = val_annotations.convert_labels(conversion_table)
val_annotations_corrected.audio_files = val_annotations_corrected.df['audio_file'].values #workaround for issue #872

train_annotations_corrected = train_annotations.convert_labels(conversion_table)
train_annotations_corrected.audio_files = train_annotations_corrected.df['audio_file'].values #workaround for issue #872

test_annotations_corrected = test_annotations.convert_labels(conversion_table)
test_annotations_corrected.audio_files = test_annotations_corrected.df['audio_file'].values #workaround for issue #872

#Step 5: pick classes to train the model on. These should occur in the annotated data
class_list = ['Hypsipetes_madagascariensis','Copsychus_albospecularis','Coracopsis_nigra','Dicrurus_forficatus','Coua_caerulea','Zosterops_maderaspatanus','Eurystomus_glaucurus','Agapornis_canus','Saxicola_torquatus','Cyanolanius_madagascarinus','Leptopterus_chabert','Nesoenas_picturatus','Coua_reynaudii','Ceblepyris_cinereus','Neodrepanis_coruscans','Philepitta_castanea','Eulemur_sp','Coua_cristata','Treron_australis']

#Step 6: create labels for fixed-duration (5 second) clips
val_labels = val_annotations_corrected.clip_labels(
  clip_duration=5,
  clip_overlap=0,
  min_label_overlap=0.25,
  class_subset=class_list
)

train_labels = train_annotations_corrected.clip_labels(
  clip_duration=5,
  clip_overlap=0,
  min_label_overlap=0.25,
  class_subset=class_list
)

test_labels = test_annotations_corrected.clip_labels(
  clip_duration=5,
  clip_overlap=0,
  min_label_overlap=0.25,
  class_subset=class_list
)

# Now you have your labels for training, validation, and testing

# -----------------
# Model training
# -----------------

#Load the pre-trained Perch2 tensorflow model
perch2_model = bmz.Perch2()

#add a 2-layer PyTorch classification head on top of the pre-trained Perch2 model
perch2_model.initialize_custom_classifier(class_list, hidden_layer_sizes=())

#embed the training/validation samples with 5 augmented variations each, 
#then fit the classification head
perch2_model.train(
  train_labels,
  val_labels,
  n_augmentation_variants=5,
  embedding_batch_size=64, 
  embedding_num_workers=4  
)
#Save model as a lightweight option that is easier to reload in different environments
import torch
state_dict_path = save_path / "classifier_state_dict.pt"
torch.save(perch2_model.network.state_dict(), state_dict_path)
print(f"Saved classifier weights to: {state_dict_path}")

# save the custom Perch2 model to a file
save_path.parent.mkdir(parents=True, exist_ok=True)
perch2_model.save(save_path)
print(f"Saved model to: {save_path}")
# later, to reload your fine-tuned Perch2 from the saved object:
# perch2_model = bmz.Perch2.load(save_path)


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

    # Keep only the class columns (and in consistent order)
    preds = preds[class_list].copy()

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
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        out_png = hist_dir / f"{filename}_{species}.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.clf()

        # Semilog histogram (y-axis)
        plt.hist(df_not[pred_col], bins=bins, alpha=0.5, label="negatives")
        plt.hist(df_pos[pred_col], bins=bins, alpha=0.5, label="positives")
        plt.legend()
        plt.xlabel("Score")
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
    out_dir=save_path,   # saves under the model folder
    filename=filename,
    batch_size=64,
)