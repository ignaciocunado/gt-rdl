#!/bin/sh
#SBATCH --job-name="global_f1_driver_top_3"
#SBATCH --partition=ewi-st,general      # Request partition.
#SBATCH --qos=short                # short (4h max), medium (1.5 dys max) or long (7 days max)
#SBATCH --time=00:02:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --cpus-per-task=4          # Request number of CPUs (threads) per task.
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=16GB                  # Request 4 GB of RAM in total
#SBATCH --mail-type=BEGIN,END,FAIL            # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --output=slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId

export APPTAINER_IMAGE="daic.sif"

# If you use GPUs
module use /opt/insy/modulefiles
module load cuda/11.8

# Run script
srun apptainer exec --nv --env-file .env -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/relbench-ignacio $APPTAINER_IMAGE python main.py --model graphormer --save_artifacts --dataset f1 --task driver-top3 --eval_freq 4 --batch_size 2048 --channels 64 --num_neighbors 50 50 --rev_mp --edge_features --dropouts 0.2 0.2 0.2

