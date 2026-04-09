# rewilding_madagascar

This repository contains the environment structure, scripts, and metadata used to train a shallow classifier on Perch2 embeddings using OpenSoundscape and labelled data from the Rewilding Madagascar project. 

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
site/
  habitat_A/
    habitat_A-recorder/
      WAV files

Filename format:
<habitat>_A-<recorder>_YYYYMMDD_HHMMSS.WAV

## Model


## How to run inference

## Output format

## Reproducibility
### Container

Inference was run using an Apptainer container built from `container/perch2.def`.

The final container used was:
`/cephyr/users/sheilaho/Alvis/perch2.sif`

Note: The `.def` file represents the intended build recipe.

apptainer inspect /cephyr/users/sheilaho/Alvis/perch2.sif

maintainer: NVIDIA CORPORATION <cudatools@nvidia.com>
org.label-schema.build-arch: amd64
org.label-schema.build-date: Monday_16_February_2026_10:21:46_CET
org.label-schema.schema-version: 1.0
org.label-schema.usage.apptainer.version: 1.4.3-1.el8
org.label-schema.usage.singularity.deffile.bootstrap: docker
org.label-schema.usage.singularity.deffile.from: tensorflow/tensorflow:latest-gpu-jupyter
org.opencontainers.image.ref.name: ubuntu
org.opencontainers.image.version: 22.04


## Funding
This project was funded by the Swedish Research Council (Registration number 2020-03239)