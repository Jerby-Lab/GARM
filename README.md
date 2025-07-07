# GAR-M
Predict Perturb-Seq with GAR-M (Gradient Aligned Regression with Multihead)
<p align="center">
  <img src="figures/GAR-MPH.jpg" width="800" title="Sine">
</p>

## Usage:

Run on the curated datasets ['norman', 'adamson', 'dixit', 'replogle_k562_essential', 'replogle_rpe1_essential'] from GEARS paper.
```python
python3 main_GARM.py --method=GARM --dataset=replogle_rpe1_essential --lr=1e-3 --K=1024 --decay=1e-8  --layers=2 --batch_size=16 
```

Run on the orginal Essential-Wide datasets ['jurkat', 'hepg2', 'k562', 'rpe1'] from Weissman lab.
```python
python3 main_GARM.py --method=GARM --dataset=rpe1 --lr=1e-3 --K=1024 --decay=1e-8  --layers=2 --batch_size=16 
```

Run cross-dataset prediction on the orginal Essential-Wide datasets ['jurkat', 'hepg2', 'k562', 'rpe1'] from Weissman lab.
```python
python3 main_GARM_cross.py --method=GARM --lr=1e-3 --K=1024 --decay=1e-8  --layers=2 --batch_size=16 
```
