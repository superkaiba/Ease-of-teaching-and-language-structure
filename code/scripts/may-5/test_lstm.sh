#!/bin/bash

#SBATCH --partition=long,main,unkillable,lab-bengioy
#SBATCH --cpus-per-task=4                             
#SBATCH --gres=gpu:v100:1                                    
#SBATCH --mem=8G                                       
#SBATCH --time=8:00:00                                
#SBATCH -o /network/scratch/t/thomas.jiralerspong/kolmogorov/slurm/slurm-%j.out

conda init bash
source activate ease_of_teaching
cd /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code


python /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code/real_main_lstm.py \
    --reset False \
    --fname $SCRATCH/kolmogorov/ease_of_teaching/results/new \
    --wandbdir $SCRATCH/kolmogorov/ease_of_teaching \
    --wandbgroup ease_of_teaching \
    --n_attributes 2 \
    --n_values 8 \
    --vocabSize 8 \
    --messageLen 2 \
    --hiddenSize 100 \
    --resetIter 10000