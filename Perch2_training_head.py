#Import all the libraries we need

import torch
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import random
from glob import glob
import sklearn

from tqdm.autonotebook import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path

#set up plotting - actually do I need this since I moved evaluation to a different file?
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=[15,5] #for large visuals
%config InlineBackend.figure_format = 'retina'

# opensoundscape transfer learning tools
from opensoundscape.ml.shallow_classifier import MLPClassifier, quick_fit, fit_classifier_on_embeddings

from sklearn.model_selection import train_test_split
from opensoundscape import BoxedAnnotations, CNN
import opensoundscape
import bioacoustics_model_zoo as bmz


#Name variables and specify files and paths

filename = 'perch2_shallow_classifier'
metadata = pd.read_csv('./data/metadata_mimer.csv') #can I do this, or do i need to specify mimer/NOBACKUP/groups/rewilding_madagascar/data/metadata_mimer.csv?
save_path = f'./mimer/NOBACKUP/groups/rewilding_madagascar/models/{filename}'

#Prepare annotation data for opensoundscape
#Step 1: split the metadata into train, validation, and test sets
val_metadata=metadata[metadata["Split"]=="validation"]
train_metadata=metadata[metadata["Split"]=="train"]
test_metadata=metadata[metadata["Split"]=="test"]
val_metadata=val_metadata.reset_index()
train_metadata=train_metadata.reset_index()
test_metadata=test_metadata.reset_index()

# Step 2: load the annotations into OpenSoundscape
raven_file_paths = val_metadata['Raven_path']
audio_file_paths = val_metadata['SoundFile_path']
val_annotations = BoxedAnnotations.from_raven_files(raven_file_paths,'species',audio_file_paths)

raven_file_paths = train_metadata['Raven_path']
audio_file_paths = train_metadata['SoundFile_path']
train_annotations = BoxedAnnotations.from_raven_files(raven_file_paths,'species',audio_file_paths)

raven_file_paths = test_metadata['Raven_path']
audio_file_paths = test_metadata['SoundFile_path']
test_annotations = BoxedAnnotations.from_raven_files(raven_file_paths,'species',audio_file_paths)

# Step 3: create a conversion table to map the original species names to the new ones we want to use for training
# Create the table with a dataframe
conversion_table = pd.DataFrame(
    {'original':['Eulemur_albifrons', 'Eulemur_fulvus'],
     'new':['Eulemur_sp', 'Eulemur_sp']}
)

# Or create the table in its own spreadsheet
#conversion_table = pd.read_csv('my_conversion_filename_here.csv')

#Step 4: correct annotations in each of the splits
val_annotations_corrected = val_annotations.convert_labels(conversion_table)
val_annotations_corrected.audio_files = val_annotations_corrected.df['audio_file'].values #workaround for issue #872
val_annotations_corrected.df.head()

train_annotations_corrected = train_annotations.convert_labels(conversion_table)
train_annotations_corrected.audio_files = train_annotations_corrected.df['audio_file'].values #workaround for issue #872
train_annotations_corrected.df.head()

test_annotations_corrected = test_annotations.convert_labels(conversion_table)
test_annotations_corrected.audio_files = test_annotations_corrected.df['audio_file'].values #workaround for issue #872
test_annotations_corrected.df.head()

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

#Load the pre-trained Perch2 tensorflow model
perch2_model = bmz.Perch2()

#add a 2-layer PyTorch classification head on top of the pre-trained Perch2 model
#how do I decide on hidden layer sizes? unclear
perch2_model.initialize_custom_classifier(classes=class_list, hidden_layer_sizes=(100,))

#embed the training/validation samples with 5 augmented variations each, 
#then fit the classification head
perch2_model.train(
  train_labels,
  val_labels,
  n_augmentation_variants=5,
  embedding_batch_size=64, #used 128 for perch embeddings
  embedding_num_workers=4  #used 0 for perch embeddings, but can speed up with more workers if you have the resources
)
# save the custom Perch2 model to a file
perch2_model.save(save_path)
# later, to reload your fine-tuned Perch2 from the saved object:
# perch2_model = bmz.Perch2.load(save_path)