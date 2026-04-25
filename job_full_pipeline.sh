#!/bin/bash
#PBS -N kd_full_pipeline
#PBS -P col772.mcs252107.course
#PBS -q standard
#PBS -l select=1:ncpus=8:ngpus=1:mem=32G:centos=skylake
#PBS -l walltime=9:00:00
#PBS -o pipeline.out
#PBS -e pipeline.err

cd $PBS_O_WORKDIR
module load compiler/cuda/12.1
module load apps/anaconda/3

pip install -r requirements.txt --quiet

bash run_distillation.sh