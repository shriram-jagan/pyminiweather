#!/bin/bash

source common.sh

export NTIMESTEPS=$(expr $NSTEPS / $OUTPUT_FREQ)

python ./make_images.py \
	--nx ${NUMPY_BASE_NX} \
	--nz ${NUMPY_BASE_NZ} \
	--ntimesteps ${NTIMESTEPS} \
	--directory ./images_numpy \
	--filename "numpy.txt" \
	--plot-vmin-vmax -22.0 22.0 \
	--plot-no-colorbar

# need to insert the multiple cpu option here

python ./make_images.py \
	--nx ${GPU_BASE_NX} \
	--nz ${GPU_BASE_NZ} \
	--ntimesteps ${NTIMESTEPS} \
	--directory ./images_gpus \
	--filename "gpus.txt" \
	--plot-vmin-vmax -22.0 22.0 \
	--plot-no-colorbar
