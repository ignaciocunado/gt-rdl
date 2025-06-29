#!/bin/sh
#SBATCH --job-name="global_avito_ad_ctr"
#SBATCH --partition=ewi-st,general      # Request partition.
#SBATCH --qos=short                # short (4h max), medium (1.5 dys max) or long (7 days max)
#SBATCH --time=00:35:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --cpus-per-task=16          # Request number of CPUs (threads) per task.
#SBATCH --gres=gpu:a40:1               # Request 1 GPU
#SBATCH --mem=64GB                  # Request 4 GB of RAM in total
#SBATCH --mail-type=BEGIN,END,FAIL            # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --output=slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId

export APPTAINER_IMAGE="daic.sif"

# If you use GPUs
module use /opt/insy/modulefiles
module load cuda/11.8

# Run script
srun apptainer exec --nv --env-file .env -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/relbench-ignacio $APPTAINER_IMAGE python main.py --model graphormer --save_artifacts --num_workers 12 --dataset avito --task ad-ctr --eval_freq 4 --batch_size 64 --epochs 15 --channels 64 --num_neighbors 50 50 --rev_mp --edge_features --dropouts 0.1 0.1 0.1 --head HeteroGNNNodeRegressionHead --seed 1

