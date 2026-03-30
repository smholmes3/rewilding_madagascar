# Preflight: build labels only (NO model training)
import pandas as pd
from pathlib import Path
from opensoundscape import BoxedAnnotations

# -----------------
# Config (matches Sam script)
# -----------------
filename = "perch2_shallow_classifier"

metadata_path = Path("/mimer/NOBACKUP/groups/rewilding_madagascar/data/metadata_mimer.csv")
save_path = Path(f"/mimer/NOBACKUP/groups/rewilding_madagascar/models/{filename}")

# -----------------
# Load metadata
# -----------------
metadata = pd.read_csv(metadata_path)
print("Loaded metadata:", len(metadata))
print(metadata[["SoundFile_path", "Raven_path", "Split"]].head(2))

# Step 1: split
val_metadata = metadata[metadata["Split"] == "validation"].reset_index(drop=True)
train_metadata = metadata[metadata["Split"] == "train"].reset_index(drop=True)
test_metadata = metadata[metadata["Split"] == "test"].reset_index(drop=True)
print("Rows train/val/test:", len(train_metadata), len(val_metadata), len(test_metadata))

# Optional: check a few paths exist (helps catch path issues fast)
def check_some(df, col, n=3):
    print(f"\nChecking {col} (first {n}):")
    for p in df[col].astype(str).head(n):
        ok = Path(p).exists()
        print(" ", ok, p)

check_some(val_metadata, "SoundFile_path")
check_some(val_metadata, "Raven_path")

# Step 2: load annotations
val_annotations = BoxedAnnotations.from_raven_files(
    val_metadata["Raven_path"], "species", val_metadata["SoundFile_path"]
)
train_annotations = BoxedAnnotations.from_raven_files(
    train_metadata["Raven_path"], "species", train_metadata["SoundFile_path"]
)
test_annotations = BoxedAnnotations.from_raven_files(
    test_metadata["Raven_path"], "species", test_metadata["SoundFile_path"]
)

# Step 3: conversion table
conversion_table = pd.DataFrame(
    {"original": ["Eulemur_albifrons", "Eulemur_fulvus"],
     "new": ["Eulemur_sp", "Eulemur_sp"]}
)

# Step 4: convert labels + workaround
val_annotations_corrected = val_annotations.convert_labels(conversion_table)
val_annotations_corrected.audio_files = val_annotations_corrected.df["audio_file"].values  # issue #872

train_annotations_corrected = train_annotations.convert_labels(conversion_table)
train_annotations_corrected.audio_files = train_annotations_corrected.df["audio_file"].values  # issue #872

test_annotations_corrected = test_annotations.convert_labels(conversion_table)
test_annotations_corrected.audio_files = test_annotations_corrected.df["audio_file"].values  # issue #872

# Step 5: class list (same as Sam script)
class_list = [
    "Hypsipetes_madagascariensis","Copsychus_albospecularis","Coracopsis_nigra",
    "Dicrurus_forficatus","Coua_caerulea","Zosterops_maderaspatanus",
    "Eurystomus_glaucurus","Agapornis_canus","Saxicola_torquatus",
    "Cyanolanius_madagascarinus","Leptopterus_chabert","Nesoenas_picturatus",
    "Coua_reynaudii","Ceblepyris_cinereus","Neodrepanis_coruscans",
    "Philepitta_castanea","Eulemur_sp","Coua_cristata","Treron_australis"
]

# Step 6: clip labels
val_labels = val_annotations_corrected.clip_labels(
    clip_duration=5, clip_overlap=0, min_label_overlap=0.25, class_subset=class_list
)
train_labels = train_annotations_corrected.clip_labels(
    clip_duration=5, clip_overlap=0, min_label_overlap=0.25, class_subset=class_list
)
test_labels = test_annotations_corrected.clip_labels(
    clip_duration=5, clip_overlap=0, min_label_overlap=0.25, class_subset=class_list
)

print("\nClip counts:")
print("Train clips:", len(train_labels))
print("Val clips:", len(val_labels))
print("Test clips:", len(test_labels))

print("\nPreflight complete: labels built successfully. No model was loaded; no training occurred.")