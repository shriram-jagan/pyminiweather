#!/bin/bash

source common.sh

export IC="collision"  # "thermal"

export RUN_SINGLE_CPU=0
export RUN_MULTI_CPU=0
export RUN_MULTI_GPU=1

# Make sure OUTPUT_FREQ divides NSTEPS

export NCPUS=4
export NGPUS=2
export FBMEM=75000
export SYSMEM=75000
export NUMAMEM=75000

echo `date`

# Single CPU
if [[ ${RUN_SINGLE_CPU} -eq 1 ]]; then
  mkdir -p images_numpy/images
  echo "Running on 1 CPU using NumPy"

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
  echo "Running on ${NCPUS} CPUs"

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

  mv PyMiniWeatherData_svars.txt images_cpus/cpus_svars.txt
  mv PyMiniWeatherData.txt       images_cpus/cpus.txt
fi

# Multiple GPUs (placeholder) 
if [[ ${RUN_MULTI_GPU} -eq 1 ]]; then
  mkdir -p images_gpus/images
  echo "Running on ${NGPUS} GPUs"

  LEGATE_TEST=1 legate \
    --gpus ${NGPUS} \
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
