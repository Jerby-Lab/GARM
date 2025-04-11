# GAR-M
Predict Perturb-Seq with GAR-M (Gradient Aligned Regression with Multihead)

## Usage:

Run on the curated datasets from GEARS paper.
```python
python3 main.py --method=GARM --dataset=replogle_rpe1_essential --lr=1e-3 --decay=5e-4 --hidden_size=128
python3 main.py --method=GEARS --dataset=replogle_rpe1_essential --lr=1e-3 --decay=5e-4 --hidden_size=128
```

Run on the orginal Essential-Wide datasets from Weissman lab.
```python
python3 main.py --method=GARM --dataset=rpe1 --lr=1e-3 --decay=5e-4 --hidden_size=128
python3 main.py --method=GEARS --dataset=rpe1 --lr=1e-3 --decay=5e-4 --hidden_size=128
```