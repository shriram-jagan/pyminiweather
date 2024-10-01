#!/bin/bash

source common.sh

export IC="collision"  # "thermal"

export RUN_SINGLE_CPU=0
export RUN_MULTI_CPU=1
export RUN_MULTI_GPU=0

# Make sure OUTPUT_FREQ divides NSTEPS
export NSTEPS=20000
export OUTPUT_FREQ=100
export DT=0.04

export N_CPUS=4

export FBMEM=45000
export SYSMEM=45000
export NUMAMEM=45000

echo `date`

# Single CPU
if [[ ${RUN_SINGLE_CPU} -eq 1 ]]; then
  mkdir -p images_numpy/images
  echo "Running on single CPU"

  ${CONDA_PREFIX}/bin/PyMiniweather.py \
    --nx ${NUMPY_BASE_NX} \
    --nz ${NUMPY_BASE_NZ} \
    --nsteps ${NSTEPS} \
    --ic-type ${IC} \
    --output-freq ${OUTPUT_FREQ} \
    --dt ${DT} \
    --verbose

  mv PyMiniWeatherData_svars.txt images_numpy/numpy_svars.txt
  mv PyMiniWeatherData.txt       images_numpy/numpy.txt
fi

# Multiple CPUs
if [[ ${RUN_MULTI_CPU} -eq 1 ]]; then
  # Simulation diverges when OMP variant is used
  # make sure to change sysmem to numamem if OMP variant is used

  mkdir -p images_cpus/images
  echo "Running on multiple CPUs"

  LEGATE_TEST=1 legate \
    --cpus ${N_CPUS} \
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

  mv PyMiniWeatherData_svars.txt images_cpus/cpus_svars.txt
  mv PyMiniWeatherData.txt       images_cpus/cpus.txt
fi

# Multiple GPUs (placeholder) 
if [[ ${RUN_MULTI_GPU} -eq 1 ]]; then
  mkdir -p images_gpus/images
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

  mv PyMiniWeatherData_svars.txt images_gpus/gpus_svars.txt
  mv PyMiniWeatherData.txt       images_gpus/gpus.txt
fi

echo `date`
