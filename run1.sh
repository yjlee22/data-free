mkdir -p output

methods=("FedAvg" "FedSMOO" "FedProx" "FedSpeed" "SCAFFOLD" "FedGamma" "FedDyn" "FedSAM" "FedLESAM" "FedWMSAM")
seeds=(0 1 2)
datasets=("BloodMNIST" "DermaMNIST" "PathMNIST")
patiences=(1 2 5 10)
declare -A NUM_CLASS=(
  ["BloodMNIST"]=8
  ["DermaMNIST"]=7
  ["PathMNIST"]=9
)
CONTAINER="$WORK/containers/custom_pytorch.sif"

for dataset in "${datasets[@]}"; do
    num_class="${NUM_CLASS[$dataset]}"
    for method in "${methods[@]}"; do
        for seed in "${seeds[@]}"; do
            for p in "${patiences[@]}"; do
                cmd="python3 train.py --local-learning-rate 0.1 --validation --fast --non-iid --dataset ${dataset} --num_class ${num_class} --pretrain --method ${method} --seed ${seed} --split-coef 0.1 --patience ${p} --threshold 0.1"

                JOB_NAME="fl_${dataset}_${method}_s${seed}_p_${p}_validation"

                sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=05:00:00
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=output/${JOB_NAME}_%j.out
#SBATCH --error=output/${JOB_NAME}_%j.err

echo "Job started at \$(date)"
echo "Node: \$(hostname)"
echo "Running: dataset=${dataset}, method=${method}, seed=${seed}, patience=${p}"

apptainer exec --bind \$WORK:\$WORK ${CONTAINER} bash -c "
  export MIOPEN_USER_DB_PATH=/tmp/miopen-\${SLURM_JOB_ID}
  export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-\${SLURM_JOB_ID}
  mkdir -p /tmp/miopen-\${SLURM_JOB_ID}
  ${cmd}
"

echo "Job finished at \$(date)"
EOF

                echo "Submitted: ${JOB_NAME}"
            done
        done
    done
done

echo "All jobs submitted. Check status with: squeue -u \$USER"