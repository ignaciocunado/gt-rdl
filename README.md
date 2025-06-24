# Graph Transformers for Relational Deep Learning
---
> This repository includes changes to [Graphormer's](https://arxiv.org/abs/2106.05234) and [FraudGT's](https://dl.acm.org/doi/10.1145/3677052.3698648) attention mechanisms to perform node-inductive tasks for Relational Deep Learning. This implementation is part of CSE3000 - Research Project
---

# Setup

## Conda

- Create a new Conda environment, and activate.
```bash
conda env create -f env.yml
conda activate relbench-env

```
- Install Pytorch and Pytorch Geometric
```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
pip install pytorch-frame sentence_transformers relbench
```

## MacOS

Create a venv, then run

```bash
source venv/bin/activate
```

Then run the following

```bash
pip install torch==2.6.0 torchvision torchaudio torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html   
pip install pytorch-frame sentence_transformers relbench wandb
```

---

# Running

```bash
python3 main.py \
    --model <local|global> \
    --dataset <f1|…> \
    --task <driver-position|…> \
    --save_artifacts \
    --num_workers <int> \
    --eval_freq <int> \
    --lr <float> \
    --epochs <int> \
    --optimiser <adam|adamW> \
    --batch_size <int> \
    --channels <int> \
    --aggr <sum|mean|max|…> \
    --num_layers <int> \
    --num_layers_pre_gt <int> \
    --num_neighbors <n₁ n₂ …> \
    --temporal_strategy uniform \
    --rev_mp \
    --port_numbering \
    --ego_ids \
    --edge_features \
    --dropouts <d_local d_global d_attn> \
    --head <HeteroGNNNodeHead|HeteroGNNNodeRegressionHead> \
    --early_stopping \
    --seed <int>
```

For example

```bash
python3 main.py \
    --model graphormer \
    --save_artifacts \
    --num_workers 12 \
    --dataset avito \
    --task ad-ctr \
    --eval_freq 4 \
    --batch_size 64 \
    --epochs 15 \
    --channels 64 \
    --num_neighbors 50 50 \
    --rev_mp \
    --edge_features \
    --dropouts 0.1 0.1 0.1 \
    --head HeteroGNNNodeRegressionHead \
    --seed 1
```

For all scripts, look at `/scripts`
