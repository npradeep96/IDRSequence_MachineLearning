#!/bin/bash
#SBATCH -J StabilityAnalysis
#SBATCH --mail-user npradeep96
#SBATCH -p sched_mit_arupc_long
#SBATCH -t 12:00:00
#SBATCH --mem-per-cpu 8000
cd "$SLURM_SUBMIT_DIR"
echo $PWD

run_program()
{
    source activate IDRBlocks
    python get_pfam_domains_and_IDRs_in_proteins.py --i $input_file --o $output_file
    conda deactivate
}

run_program

