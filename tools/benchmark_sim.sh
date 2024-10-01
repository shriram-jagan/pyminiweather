#!/bin/bash

source common.sh

export IC="collision"  # "thermal"

export RUN_SINGLE_CPU=1
export RUN_MULTI_CPU=1
export RUN_MULTI_GPU=1

export NSTEPS=200
export OUTPUT_FREQ=-1
export DT=0.04

export NCPUS=4
export FBMEM=45000
export SYSMEM=45000
export NUMAMEM=45000

echo `date`

# Single CPU
if [[ ${RUN_SINGLE_CPU} -eq 1 ]]; then
  echo "Running on single CPU"
  ${CONDA_PREFIX}/bin/PyMiniweather.py \
    --nx ${NUMPY_BASE_NX} \
    --nz ${NUMPY_BASE_NZ} \
    --nsteps ${NSTEPS} \
    --ic-type ${IC} \
    --output-freq ${OUTPUT_FREQ} \
    --dt ${DT} \
    --verbose
fi

# Multiple CPUs
if [[ ${RUN_MULTI_CPU} -eq 1 ]]; then
  echo "Running on multiple CPUs"
  LEGATE_TEST=1 legate \
    --cpus ${NCPUS} \
    --sysmem ${SYSMEM} \
    --eager-alloc-percentage 10 \
    ${CONDA_PREFIX}/bin/PyMiniweather.py \
    --nx ${CPU_BASE_NX} \
    --nz ${CPU_BASE_NZ} \
    --nsteps ${NSTEPS} \
    --ic-type ${IC}  \
    --output-freq ${OUTPUT_FREQ} \
    --dt ${DT} \
    --verbose 
fi

# Multiple GPUs (placeholder) 
if [[ ${RUN_MULTI_GPU} -eq 1 ]]; then
  echo "Running on multiple GPUs"
  LEGATE_TEST=1 legate \
    --gpus 1 \
    --fbmem ${FBMEM}\
    --eager-alloc-percentage 10 \
    ${CONDA_PREFIX}/bin/PyMiniweather.py \
    --nx ${GPU_BASE_NX} \
    --nz ${GPU_BASE_NZ} \
    --nsteps ${NSTEPS} \
    --ic-type ${IC} \
    --output-freq ${OUTPUT_FREQ} \
    --dt ${DT} \
    --verbose
fi

echo `date`
