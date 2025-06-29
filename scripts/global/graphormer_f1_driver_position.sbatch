#!/bin/sh
#SBATCH --job-name="global_f1_driver_position"
#SBATCH --partition=ewi-st,general      # Request partition.
#SBATCH --qos=short                # short (4h max), medium (1.5 dys max) or long (7 days max)
#SBATCH --time=00:15:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --cpus-per-task=14          # Request number of CPUs (threads) per task.
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
srun apptainer exec --nv --env-file .env -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/relbench-ignacio $APPTAINER_IMAGE python main.py --model graphormer --num_workers 12 --save_artifacts --dataset f1 --task driver-position --epochs 12 --optimiser adamW --eval_freq 2 --lr 0.0013 --batch_size 64 --channels 128 --num_layers_pre_gt 2 --num_neighbors 60 60 --rev_mp --port_numbering --dropouts 0.4 0.08 0.17 --head HeteroGNNNodeRegressionHead --seed 1

