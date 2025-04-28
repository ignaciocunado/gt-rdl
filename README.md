## Setup

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
---

## MacOS setup

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