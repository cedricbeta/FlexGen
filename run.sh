#!/usr/bin/env zsh
#SBATCH --job-name=flexgen
#SBATCH --partition=instruction
#SBATCH --time=00-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# SBATCH --array=0-3
# SBATCH --gres=gpu:2 -c=3 -N=4
#SBATCH -o task-%j.out -e task-%j-error.err
#SBATCH --mem=32768

conda env list
source virtual_env/bin/activate

pip --version
python --version

pip install -e .
# python hw_optim/auto_config_optimizer.py --model facebook/opt-6.7b --prompt-len 256 --gen-len 32 --gpu-mem 8 --cpu-mem 32 --nvme-mem 1500
python hw_optim/auto_config_optimizer.py --model facebook/opt-1.3b --prompt-len 128 --gen-len 32 --gpu-mem 2 --cpu-mem 4 --nvme-mem 1500

echo "Done"
python -m flexgen.flex_opt --model facebook/opt-6.7b --cut-gen-len 8 --prompt-len 256 --gen-len 32 --overlap --cpu-cache-compute --attn-sparsity 1 --log-file temp.log --percent 63 37 8 92 0 100
# python -m flexgen.flex_opt --model facebook/opt-1.3b


# --model facebook/opt-1.3b --gpu-batch-size 16 --percent 50 50 100 0 100 0 --cut-gen-len 8

# for i in {10..29}; do echo "Run $i"; ./task2 $((2**i)) 128 1024; done;
# for i in {10..29}; do echo "Run $i"; ./task2 $((2**i)) 128 512; done;

