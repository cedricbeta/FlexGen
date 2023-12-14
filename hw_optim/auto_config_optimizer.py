import sys
import os
import subprocess
import argparse
from opt_model import OptModelConfig, solve, solve_lp
from flexgen.opt_config import get_opt_config
from flexgen.utils import GB
from flexgen.flex_opt import Policy

from results_collector import store_entry


def test_policy(args, policy: Policy):
    cmd = f"python -m flexgen.flex_opt --model {args.model} --cut-gen-len 8"
    if args.prompt_len:
        cmd += f" --prompt-len {args.prompt_len}"
    if args.gen_len:
        cmd += f" --gen-len {args.gen_len}"
    if args.gbs:
        cmd += f" --gpu-batch-size {args.gbs}"
    if args.num_gb:
        cmd += f" --num-gpu-batches {args.num_gb}"

    if policy.overlap:
        cmd += f" --overlap"
    if policy.sep_layer:
        cmd += f" --sep-layer"
    if policy.pin_weight:
        cmd += f" --pin-weight"
    if policy.cpu_cache_compute:
        cmd += f" --cpu-cache-compute"
    if policy.attn_sparsity:
        cmd += f" --attn-sparsity {policy.attn_sparsity}"
    if policy.compress_weight:
        cmd += f" --compress-weight {policy.compress_weight}"
    if policy.compress_cache:
        cmd += f" --compress-cache {policy.compress_cache}"

    def run_benchmark(id, percents, delta):
        # return throughput, status, percents
        # status: 0 - success, 1 - this optimization is not possible
        wg, wc, cg, cc, ag, ac = percents
        # Transfering weights
        if wg + delta + wc <= 100:  # disk has at least delta weights
            wg += delta
        elif wg + delta <= 100:  # disk has less than delta weights
            wg += delta
            wc = 100 - wg
        if wg + delta + wc <= 100:  # disk has at least delta weights
            wc += delta
        else:  # disk has less than delta weights
            wc = 100 - wg
        # Transfering cache
        if cg + delta + cc <= 100:  # disk has at least delta cache
            cg += delta
        elif cg + delta <= 100:  # disk has less than delta cache
            cg += delta
            cc = 100 - cg
        if cg + delta + cc <= 100:  # disk has at least delta cache
            cc += delta
        else:  # disk has less than delta cache
            cc = 100 - cg
        new_percents = [int(wg), int(wc), int(cg), int(cc), int(ag), int(ac)]
        if (
            delta != 0 and # delta is zero only for the first run
            sum([n != o for n, o in zip(new_percents, percents)]) == 0
        ):  # nothing changed
            return None, 1, percents
        log_file = f"logs/auto_optim_{os.environ['SLURM_JOB_ID']}_{id}.log"
        command = cmd + f" --log-file {log_file}"
        command += " --percent " + " ".join(map(str, new_percents))
        if os.path.exists(log_file):
            os.remove(log_file)
        print(f'Running command:"{command}"', flush=True)

        output = subprocess.run(command, shell=True, capture_output=True)
        # if output.stderr:
        #     print(
        #         "CUDA out-of-memory Error",
        #         # "CUDA out-of-memory Error encountered: {}".format(output.stderr.decode("utf-8")),
        #         file=sys.stderr,
        #         flush=True,
        #     )
        #     return None, 1, new_percents
        # if output.stdout:
        #     lines = output.stdout.decode("utf-8").split("\n")
        #     for line in lines:
        #         if "total throughput: " in line:
        #             throughput = float(line.split()[-2])
        #     print(f"Derived throughput: {throughput}", file=sys.stderr, flush=True)
        #     return throughput, 0, new_percents
        # print("Invalid output", output, file=sys.stderr, flush=True)
        
        throughput = None
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f.readlines():
                    if "total throughput: " in line:
                        throughput = float(line.split()[-2])
        if throughput is not None:
            print(f"Derived throughput: {throughput}", file=sys.stderr, flush=True)
            return throughput, 0, new_percents
        print("Log file not found", file=sys.stderr, flush=True)
        return None, 1, percents

    def better_result(result1, result2):
        return result1[0] > result2[0]

    def get_percents(policy: Policy):
        wg, wc, cg, cc, ag, ac = (
            policy.w_gpu_percent * 100,
            policy.w_cpu_percent * 100,
            policy.cache_gpu_percent * 100,
            policy.cache_cpu_percent * 100,
            policy.act_gpu_percent * 100,
            policy.act_cpu_percent * 100,
        )
        # workaround for ac and ag - line 794 in flex_opt.py
        ad = 100 - ag - ac
        max_act_percent = max(ag, ac, ad)
        if max_act_percent == ag:
            ag = 100
            ac = 0
        elif max_act_percent == ac:
            ag = 0
            ac = 100
        else:
            ag = 0
            ac = 0
        return [int(wg), int(wc), int(cg), int(cc), int(ag), int(ac)]

    percents = get_percents(policy)
    id = 0
    base_results = run_benchmark(id, percents, 0)
    print(f"Base results = {base_results}")
    best_results = base_results
    delta, min_delta = 8, 2
    while True:
        id += 1
        print(f"Running while loop for ID={id}, Delta={delta}", flush=True)
        curr_results = run_benchmark(id, best_results[2], delta)
        if curr_results[1] == 1:  # this optimization was not possible
            delta = delta / 2  # reduce delta
        elif curr_results[1] == 0 and better_result(
            curr_results, best_results
        ):  # check result with best result
            print(f"Improved results = {curr_results}")
            best_results = curr_results
        else:  # no more optimization was observed
            break
        if delta < min_delta:  # delta is too small
            break

    print(f"Best results = {best_results}")
    print(
        f"Optimal run config with given constraints: \"{cmd} --percent {' '.join(map(str, best_results[2]))}\""
    )
    return base_results, best_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-175b")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--gpu-mem", type=int, default=15)
    parser.add_argument("--cpu-mem", type=int, default=200)
    parser.add_argument("--nvme-mem", type=int, default=1500)

    parser.add_argument("--gbs", "--gpu-batch-size", type=int)
    parser.add_argument("--num-gb", "--num-gpu-batches", type=int)
    parser.add_argument("--compress-w", action="store_true")

    args = parser.parse_args()

    config = OptModelConfig()

    opt_config = get_opt_config(args.model)
    config.l = opt_config.num_hidden_layers
    config.h1 = opt_config.hidden_size
    config.h2 = opt_config.ffn_embed_dim
    config.nh = opt_config.n_head

    config.s = args.prompt_len
    config.n = args.gen_len

    config.gmem = args.gpu_mem * GB
    config.cmem = args.cpu_mem * GB
    config.nmem = args.nvme_mem * GB

    best_policy, max_throughput = solve(config, solve_lp, vars(args))
    print(f"theoretical throughput: {max_throughput:.2f} token/s")
    print("Identified best_policy:", best_policy)
    base_results, best_results = test_policy(args, best_policy)
    log_entry = {
        'job_id': os.environ['SLURM_JOB_ID'],
        'cpu_threads': os.environ['SLURM_CPUS_PER_TASK'], 
        'gpus': os.environ['SLURM_GPUS_ON_NODE'],
        'model_name': os.environ['MODEL_NAME'],
        'gpu_mem': os.environ['GPU_MEM_CONTRAINT'],
        'cpu_mem': os.environ['CPU_MEM_CONTRAINT'],
        'base_throughput': base_results[0],
        'improved_throughput': best_results[0],
        'base_allocations': base_results[2],
        'improved_allocations': best_results[2],
                }
    print('log_entry', log_entry)
    store_entry(log_entry)

