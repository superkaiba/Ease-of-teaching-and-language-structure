#!/bin/bash

#SBATCH --partition=long,main,unkillable,lab-bengioy
#SBATCH --cpus-per-task=4                             
#SBATCH --gres=gpu:v100:1                                    
#SBATCH --mem=8G                                       
#SBATCH --time=8:00:00                                
#SBATCH -o /network/scratch/t/thomas.jiralerspong/kolmogorov/slurm/slurm-%j.out

python /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code/main.py \
    --reset True \
    --fname /home/mila/t/thomas.jiralerspong/kolmogorov/scratch/kolmogorov/results \
    --n_attributes 4 \
    --n_values 10 \
    --vocabSize 10 \
    --messageLen 4 \
    --hiddenSize 100 \
    --resetIter 5000


python /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code/main.py \
    --reset True \
    --fname /home/mila/t/thomas.jiralerspong/kolmogorov/scratch/kolmogorov/results \
    --n_attributes 4 \
    --n_values 10 \
    --vocabSize 10 \
    --messageLen 4 \
    --hiddenSize 100 \
    --resetIter 10000

python /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code/main.py \
    --reset True \
    --fname /home/mila/t/thomas.jiralerspong/kolmogorov/scratch/kolmogorov/results \
    --n_attributes 4 \
    --n_values 10 \
    --vocabSize 10 \
    --messageLen 4 \
    --hiddenSize 100 \
    --resetIter 20000

# python /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code/main.py \
#     --reset False \
#     --fname /home/mila/t/thomas.jiralerspong/kolmogorov/scratch/kolmogorov/results \
#     --n_attributes 4 \
#     --n_values 10 \
#     --vocabSize 10 \
#     --messageLen 4 \
#     --hiddenSize 500 \