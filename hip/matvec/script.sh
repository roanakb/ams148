#!/bin/bash
#SBATCH --job-name=matvec
#SBATCH --output=matvec%j.out
#SBATCH --error=matvec%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rdbavisk@ucsc.edu
#SBATCH --partition=gpuq

srun matvec.exe

