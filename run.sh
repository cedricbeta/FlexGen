#!/usr/bin/env zsh
#SBATCH --job-name=flexgen
#SBATCH --partition=instruction
#SBATCH --time=00-02:00:00
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=2
# SBATCH --nodes=1
# SBATCH --gres=gpu:1
# SBATCH --array=0-3
#SBATCH --gres=gpu:1 --cpus-per-task=4
#SBATCH -o logs/task-%j.out -e logs/task-%j-error.err
#SBATCH --mem=32768

conda env list
source virtual_env/bin/activate

export GPU_MEM_CONTRAINT=2
export CPU_MEM_CONTRAINT=32
export MODEL_NAME="opt-1.3b"
# export MODEL_NAME="opt-6.7b"
# printenv
# echo "{ 'job_id': '$SLURM_JOB_ID', 'cpu_threads': $SLURM_CPUS_PER_TASK, 'gpus': $SLURM_GPUS_ON_NODE, 'model_name': '$MODEL_NAME', 'gpu_mem': $GPU_MEM_CONTRAINT, 'cpu_mem': $CPU_MEM_CONTRAINT }"

pip install -e .
python hw_optim/auto_config_optimizer.py --model facebook/$MODEL_NAME --prompt-len 256 --gen-len 32 --gpu-mem $GPU_MEM_CONTRAINT --cpu-mem $CPU_MEM_CONTRAINT --nvme-mem 1500
# python hw_optim/auto_config_optimizer.py --model facebook/opt-1.3b --prompt-len 128 --gen-len 32 --gpu-mem 2 --cpu-mem 4 --nvme-mem 1500

echo "Done"
# python -m flexgen.flex_opt --model facebook/opt-6.7b --cut-gen-len 8 --prompt-len 256 --gen-len 32 --overlap --cpu-cache-compute --attn-sparsity 1 --log-file temp.log --percent 63 37 8 92 0 100
# python -m flexgen.flex_opt --model facebook/opt-1.3b


# --model facebook/opt-1.3b --gpu-batch-size 16 --percent 50 50 100 0 100 0 --cut-gen-len 8

# for i in {10..29}; do echo "Run $i"; ./task2 $((2**i)) 128 1024; done;
# for i in {10..29}; do echo "Run $i"; ./task2 $((2**i)) 128 512; done;

