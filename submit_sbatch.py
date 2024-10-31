import os
import subprocess

# SBATCH parameters
time = "24:00:00"
account = "p200234"
cpus_per_task = "128"
qos = "default"
partition = "gpu"
gres = "gpu:4"
output_file_pattern = "logs/dcn_cifar_{}.log"  # Pattern to include seed in filename

# Base program arguments without the seed
base_program_args = [
    "dcn_cifar",
    "--gpu",
    "0",
    "1",
    "2",
    "3",
    "--dataset_name",
    "cifar10",
    "--ae_lr",
    "0.003",
    "--ae_weight_decay",
    "0.0001",
    "--byol_augmentation",
    "False",
]

# Seeds for iteration
seeds = ["214"]


def create_sbatch_script(filename, seed):
    with open(filename, "w") as file:
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH --time={time}\n")
        file.write(f"#SBATCH --account={account}\n")
        file.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        file.write(f"#SBATCH --qos={qos}\n")
        file.write(f"#SBATCH --partition={partition}\n")
        file.write(f"#SBATCH --gres={gres}\n")
        file.write(f"#SBATCH --output={output_file_pattern.format(seed)}\n")  # Use seed in output filename
        file.write("source ~/.bashrc\n")
        file.write("mamba activate brb\n")
        file.write("export CUBLAS_WORKSPACE_CONFIG=:4096:8\n")
        file.write("export CUDA_VISIBLE_DEVICES=0,1,2,3\n")
        # Include seed in program arguments
        program_args_with_seed = [*base_program_args, "--seed", str(seed)]
        file.write("python runner.py " + " ".join(program_args_with_seed) + "\n")


def submit_job(script_filename):
    submit_command = f"sbatch {script_filename}"
    subprocess.run(submit_command, shell=True)
    os.remove(script_filename)


for seed in seeds:
    script_filename = f"submit_job_seed_{seed}.sbatch"
    create_sbatch_script(script_filename, seed)
    submit_job(script_filename)
