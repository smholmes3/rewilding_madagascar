# rewilding_madagascar

This repository contains the environment structure, scripts, and metadata used to train a shallow classifier on Perch2 embeddings using OpenSoundscape and labelled data from the Rewilding Madagascar project. The goal of this pipeline is to generate species-level acoustic predictions and candidate detections for manual validation.


## Repository structure
metadata/   - file inventory and recorder list  
scripts/    - training, evaluation, and inference scripts  
slurm/      - job submission scripts  
container/  - Apptainer definition  
environment/ - package requirements  
results/    - (not tracked) output files  
data/       - (not tracked) input files  
archive/    - alternative models and experimental scripts  
logs/       - logs from training and inference runs  
tools/      - helper scripts and notebooks  


## Data availability
The raw audio data used in this project are currently stored on HPC infrastructure (Mimer/Alvis) and are not yet publicly available due to their large size.

We are actively working toward making the dataset openly available. The intention is to release the data through a suitable long-term repository once project outputs are finalized.

In the meantime, this repository documents:
- the expected data structure
- file naming conventions
- metadata formats

This should allow reproduction of the workflow once data access is available.


## Data structure
Expected directory structure:
```
site/
  habitat_A/
    habitat_A-recorder/
      WAV files
```

Filename format:
<habitat>_A-<recorder>_YYYYMMDD_HHMMSS.WAV


## Model
The Perch2 shallow classifier was trained using the `scripts/Perch2_train_shallow_classifier.py` script and evaluated using the `scripts/Perch2_eval_only_from_weights.py` script. Both training and evaluation were run as Slurm jobs, using the structure in `slurm/eval_perch2`. 

The final trained model weights are not currently included in this repository.
At the time of inference, the weights file was stored separately on HPC storage as:

/mimer/NOBACKUP/groups/rewilding_madagascar/models/<run_name>/classifier_state_dict.pt

The code in this repository is public, but trained model weights will be shared separately at a later stage.


## Environment

Inference was run using an Apptainer container built from `container/perch2.def`.

The final container used was:
`/cephyr/users/sheilaho/Alvis/perch2.sif`

Note: The `.def` file represents the intended build recipe.

Upon inspection, the container looks as follows: 

maintainer: NVIDIA CORPORATION <cudatools@nvidia.com>
org.label-schema.build-arch: amd64
org.label-schema.build-date: Monday_16_February_2026_10:21:46_CET
org.label-schema.schema-version: 1.0
org.label-schema.usage.apptainer.version: 1.4.3-1.el8
org.label-schema.usage.singularity.deffile.bootstrap: docker
org.label-schema.usage.singularity.deffile.from: tensorflow/tensorflow:latest-gpu-jupyter
org.opencontainers.image.ref.name: ubuntu
org.opencontainers.image.version: 22.04

Further details on packages are in the `environment/requirements.txt` file


## How to run inference
We ran a slurm array for inference, using the script `scripts/run_perch2_inference.py` in combination with `scripts/run_perch2_inference_array`. I used the slurm file `slurm/perch2_inference_array`. Each array task processes a single recorder, using `metadata/recorders.txt` to map SLURM_ARRAY_TASK_ID to recorder_key.

Example command: sbatch slurm/perch2_inference_array

## Output format
The output goes to a results folder, with two subfolders. The predictions_by_recorder folder contains the raw output csv files (one per recorder). Each file contains metadata columns followed by species probability columns. 

filepath,	start_time,	end_time,	Hypsipetes_madagascariensis,	Copsychus_albospecularis,	Coracopsis_nigra,	Dicrurus_forficatus,	Coua_caerulea,	Zosterops_maderaspatanus,	Eurystomus_glaucurus,	Agapornis_canus,	Saxicola_torquatus,	Cyanolanius_madagascarinus,	Leptopterus_chabert,	Nesoenas_picturatus,	Coua_reynaudii,	Ceblepyris_cinereus,	Neodrepanis_coruscans,	Philepitta_castanea,	Eulemur_sp,	Coua_cristata,	Treron_australis,	filename,	site,	habitat_code,	recorder_id,	recorder_key,	date,	datetime_start.  

The filename is predictions_<site>_<habitat>_<A>_<recorder>.csv

The review_top20_by_recorder folder contains one file per recorder, with only the top 20 scores per species per day for that recorder. Columns are: 
filepath,	filename,	site,	habitat_code,	recorder_id,	recorder_key,	date,	datetime_start,	start_time,	end_time,	species,	score,	rank_within_day

The filename is top20_review_<site>_<habitat>_<A>_<recorder>.csv


## Funding
This project was funded by the Swedish Research Council (Registration number 2020-03239)