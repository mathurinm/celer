# CELER


# Installation
Clone the repository:

```
$git clone https://github.com/mathurinm/CELER.git
$cd CELER/
$conda env create --file environment.yml
$source activate celer-env
$pip install --no-deps -e .
```

# Demos
```
$source activate celer-env
$cd CELER/
$ipython
%run examples/example_finance_path.py
```

# Dependencies
All dependencies are in  ```./environment.yml```

# Cite
If you use this code, please cite [this paper](https://arxiv.org/abs/1802.07481):

Mathurin Massias, Alexandre Gramfort and Joseph Salmon

Dual Extrapolation for Faster Lasso Solvers

Arxiv preprint, 2018
