# celer

[![](https://travis-ci.org/mathurinm/celer.svg?branch=master)](https://travis-ci.org/mathurinm/celer/)
[![](https://codecov.io/gh/mathurinm/celer/branch/master/graphs/badge.svg?branch=master)](https://codecov.io/gh/mathurinm/celer)


Fast algorithm to solve the Lasso with dual extrapolation

# Documentation

Please visit [https://mathurinm.github.io/celer/](https://mathurinm.github.io/celer/) for the latest version of the documentation.

# Install the released version

Assuming you have a working Python environment, e.g. with Anaconda you can [install CELER with pip](https://pypi.python.org/pypi/celer/).

From a console or terminal install CELER with pip:

	pip install -U celer

To setup a fully functional environment we recommend you download this [conda environment](https://raw.githubusercontent.com/mathurinm/celer/master/environment.yml) and install it with:

	conda env create --file environment.yml

# Install the development version

From a console or terminal clone the repository and install CELER:

	git clone https://github.com/mathurinm/celer.git
	cd celer/
	conda env create --file environment.yml
	source activate celer-env
	pip install --no-deps -e .


# Demos & Examples

You find on the documentation examples on the [Leukemia dataset](https://mathurinm.github.io/celer/auto_examples/plot_leukemia_path.html) (comparison with scikit-learn)
and on the [Finance/log1p dataset](https://mathurinm.github.io/celer/auto_examples/plot_finance_path.html) (more significant, but it takes times to download the data, preprocess it, and compute the path).

# Dependencies

All dependencies are in  `./environment.yml`

# Cite

If you use this code, please cite:

	Mathurin Massias, Alexandre Gramfort and Joseph Salmon
	Dual Extrapolation for Faster Lasso Solvers
	Arxiv preprint, 2018

Link: [https://arxiv.org/abs/1802.07481](https://arxiv.org/abs/1802.07481)
