# GARM
Official implementation for GARM method from the paper: **Pairwise Regression Enhances the Learning of Genetic Perturbation Transcriptional Effects for Experimental Design**.

<p align="center">
  <img src="figures/GARM.jpg" width="800" title="Overview for GARM">
</p>

## Quick Start:
Clone the github to your developement space.
```bash
git clone git@github.com:DixianZhu/GARM.git
```
Install the anaconda environment for dependencies of the code (with your conda on). This step can be skipped if you already have an environment with necessary packages such as torch, torch_geometric, scanpy, etc. Then, activate the working environment.
```bash
conda env create -f environment.yml
conda activate GARM
```

## Usage:
Run GARM on within-dataset Prediction on Unseen Single-Gene Perturbation Transcriptional Responses (on 'Curated K562' dataset).
```python
python3 main_GARM.py --dataset=curated_k562 --lr=1e-3 --decay=1e-8 --K=1024 --layers=2 --batch_size=16 
```

Run GARM on Cross-Dataset Prediction on Single-Gene Perturbation Transcriptional Responses, where the Essential-Wide datasets ['jurkat', 'hepg2', 'k562', 'rpe1'] are taken from Nadig et al. and Replogle et al.
```python
python3 main_GARM_cross.py --lr=1e-3 --decay=1e-6 --K=1024 --layers=2 --batch_size=16 
```

