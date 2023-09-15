#!/bin/bash
#SBATCH -J IDRBlocks
#SBATCH --mail-user npradeep96
#SBATCH -p sched_mit_arupc_long
#SBATCH -t 12:00:00
#SBATCH --mem-per-cpu 8000
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8

cd "$SLURM_SUBMIT_DIR"
echo $PWD

run_program()
{
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/IDRBlocks/lib
    conda activate IDRBlocks
    python generate_embeddings.py --i '../data/filtered_sequence_data/filtered_sequences_linker_45_stitch_12_filtered.fasta' --o '../data/filtered_sequence_data/filtered_sequences_linker_45_stitch_12_filtered_embeddings.pkl'
    conda deactivate
    unset LD_LIBRARY_PATH
}

run_program

