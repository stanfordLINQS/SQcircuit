<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="pics/dark_logo_sq.png">
  <source media="(prefers-color-scheme: light)" srcset="pics/light_logo_sq.png">
  <img alt="Logo image" src="pics/dark_logo_sq.png" width="350" height="auto">
</picture></div>

# SQcircuit: Superconducting Quantum Circuit Analyzer
[**What is SQcircuit?**](#What-is-SQcircuit?)
| [**Installation**](#Installation)
| [**Documentation**](#Documentation)
| [**Examples**](#Examples)
| [**Contribution**](#Contribution)

[![license](https://img.shields.io/badge/license-New%20BSD-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![codecov](https://codecov.io/gh/stanfordLINQS/SQcircuit/branch/main/graph/badge.svg?token=6FT6L9ZPHP)](https://codecov.io/gh/stanfordLINQS/SQcircuit)
[![Conda-Forge Badge](https://anaconda.org/conda-forge/sqcircuit/badges/downloads.svg)](https://anaconda.org/conda-forge/sqcircuit)

> ⚠️ **Note:** SQcircuit is compatible with QuTip versions 5.0 and above.

## What is SQcircuit?

SQcircuit is an open-source Python package designed to facilitate the 
analysis and optimization of arbitrary superconducting quantum circuits. 
Developed by researchers at Stanford University, SQcircuit provides a 
comprehensive framework to model, analyze, and optimize quantum circuits by 
constructing and diagonalizing their Hamiltonian from physical descriptions 
and efficient basis construction. This package supports the calculation of 
key circuit properties such as energy spectra, coherence times, transition 
matrix elements, coupling operators, and phase coordinate representations of 
eigenfunctions. With the integration of automatic differentiation 
capabilities using PyTorch, SQcircuit enables efficient computation of 
gradients for all the mentioned properties and custom-made loss functions, 
making it a powerful tool for optimizing superconducting quantum circuits.

The detailed theory behind the SQcircuit core code and an introduction to 
the library's functionalities are provided in the following paper:

> Taha Rajabzadeh, Zhaoyou Wang, Nathan Lee, Takuma Makihara, Yudan Guo, 
> Amir H. Safavi-Naeini,<br>
> *Analysis of arbitrary superconducting quantum circuits accompanied by a 
> Python package: SQcircuit*,<br>
> Quantum 7, 1118,<br>
> https://quantum-journal.org/papers/q-2023-09-25-1118/

## Installation
For Python above 3.6, SQcirucit can be simply installed via Conda:
```
conda install -c conda-forge sqcircuit
```
Alternatively, installation via pip is also provided. 
(Note that installing pip under Conda environment is not recommended.)
```
pip install SQcircuit
```

## Documentation
The documentation of the SQcircuit is provided at:
[sqcircuit.org](https://sqcircuit.org)

## Examples
To show the potential of SQcircuit for analyzing the arbitrary superconducting 
quantum circuits, we have provided variety of examples from state-of-the-art 
circuits in the literature at:

[examples.sqcircuit.org](https://docs.sqcircuit.org/examples.html)

The source of Jupyter notebook examples can be found at:

https://github.com/stanfordLINQS/SQcircuit-examples
## Contribution
You are very welcome to contribute to SQcircuit development by forking this 
repository and sending pull requests, or filing bug reports at the 
[issues](https://github.com/stanfordLINQS/SQcircuit/issues) page. 
All code contributions are acknowledged in the contributors' section in 
the documentation.

