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
Download the ancillary files, such as, GO (Gene Ontology), scGPT, GenePT based gene-feature, the cross-dataset in pseudo-bulk leve, gene signature files, etc. from [figshare](https://doi.org/10.6084/m9.figshare.29947694.v2). Unzip data.zip under the GARM code folder.

**[Optional]** For within-dataset experiment, if you want to run on the original HepG2, Jurkat, K562 and RPE1 datasets, please download the 'xxx_raw_singlecell_01.h5ad' files from their original paper ([GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667) and [figshare](https://doi.org/10.25452/figshare.plus.20029387.v1)) to the data folder (under 'GARM/data/xxx_essential_sc/').

## Usage:
Run GARM on within-dataset Prediction on Unseen Single-Gene Perturbation Transcriptional Responses (on 'Curated K562' dataset).
```python
python3 main_GARM.py --dataset=curated_k562 --lr=1e-3 --decay=1e-8 --K=1024 --layers=2 --batch_size=16 
```

Run GARM on Cross-Dataset Prediction on Single-Gene Perturbation Transcriptional Responses, where the Essential-Wide datasets ['jurkat', 'hepg2', 'k562', 'rpe1'] are taken from Nadig et al. and Replogle et al.
```python
python3 main_GARM_cross.py --lr=1e-3 --decay=1e-6 --K=1024 --layers=2 --batch_size=16 
```

Please explore linear.py, linear_cross.py, main_sc.py, main_GAR.py for running other compared methods.
## Citation:
If you find GARM useful in your work, please cite the following [placeholder](https://arxiv.org/abs/2402.06104):
```
@misc{Placeholder,
      title={Placeholder}, 
      author={Dixian Zhu and Livnat Jerby},
      year={2025},
}
```

## Contact:
For any inquiries on this repository please feel free to post these under "issues" or reach out to Dixian Zhu (dixian-zhu@stanford.edu) and Livnat Jerby (ljerby@stanford.edu).
