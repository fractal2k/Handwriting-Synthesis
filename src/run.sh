#!/bin/bash
### Job Name
#PBS -N handwritingsynthesis
#PBS -l walltime=01:00:00
#PBS -q batch
#PBS -o $PBS_JOBNAME.out
#PBS -e $PBS_JOBNAME.err
#PBS -l select=1:ncpus=1:gpu=2
### Send email on abort, begin and end
#PBS -m abe
### Specify mail recipient
#PBS -M sujayrokade@raitcompshpc

cd $PBS_O_WORKDIR

#conda activate dl-env

### Run the executable
/home/sujayrokade/miniconda3/envs/dl-env/bin/python /home/sujayrokade/hsynthesis/src/train.py -e $1 -l $2 -c $3
