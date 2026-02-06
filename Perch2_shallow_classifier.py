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
perch2_model.initialize_custom_classifier(classes=class_list, hidden_layer_sizes=(100,))

#embed the training/validation samples with 5 augmented variations each, 
#then fit the classification head
perch2_model.train(
  train_labels,
  val_labels,
  n_augmentation_variants=5,
  embedding_batch_size=64, 
  embedding_num_workers=4  
)
# save the custom Perch2 model to a file
save_path.parent.mkdir(parents=True, exist_ok=True)
perch2_model.save(save_path)
print(f"Saved model to: {save_path}")
# later, to reload your fine-tuned Perch2 from the saved object:
# perch2_model = bmz.Perch2.load(save_path)