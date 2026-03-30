# rewilding_madagascar

## Container

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

