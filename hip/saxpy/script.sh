#!/bin/bash
#SBATCH --job-name=saxpy
#SBATCH --output=saxpy%j.out
#SBATCH --error=saxpy%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rdbavisk@ucsc.edu
#SBATCH --partition=gpuq

srun ./saxpy.exe

