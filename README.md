# PIBS: Permutational Invariance Block Solver


This python package implements an efficient, numerically exact method for solving the dynamics of the dissipative Tavis-Cummings model with local loss and dephasing processes (no gain!). The method exploits the weak permutation symmetry and the weak U(1) symmetry of the Lindblad master equation.

This code has been used in the publication [1].


## Installation
PIBS can be installed using pip:
```
$ python3 -m pip install pibs
```

## Example/Tutorial
An example for using PIBS is given in `tutorials/example.py`, and a jupyter-notebook with more detailed explanation in `tutorials/pibs_example.ipynb`


- **[1]** L. Freter, P. Fowler-Wright, J. Cuerda, B. W. Lovett, J. Keeling, and P. Törmä, ‘Theory of dynamical superradiance in organic materials’, arXiv: arXiv:2509.03067 (https://doi.org/10.48550/arXiv.2509.03067) (2025)  

