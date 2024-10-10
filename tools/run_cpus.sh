#!/bin/bash

#export IC="collision"  # "thermal"
export IC="density-current"

export RUN_SINGLE_CPU=0
export RUN_MULTI_CPU=1
export RUN_MULTI_GPU=0

export NUMPY_BASE_NX=400
export NUMPY_BASE_NZ=200

export CPU_BASE_NX=400
export CPU_BASE_NZ=200

export GPU_BASE_NX=400
export GPU_BASE_NZ=200

export NSTEPS=1
export OUTPUT_FREQ=-1
export DT=0.1

export N_CPUS=4

export FBMEM=45000
export SYSMEM=45000
export NUMAMEM=45000

echo `date`

move_file_if_exists() {
  local file_path="$1"
  local new_file_path="$2"
  if [ -f "$file_path" ]; then
    mv "$file_path" "$new_file_path" 
  else
    echo "File $file_path does not exist."
  fi
}


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
  # Simulation diverges when OMP variant is used
  # make sure to change sysmem to numamem if OMP variant is used

  echo "Running on ${N_CPUS} CPUs"

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
