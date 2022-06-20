<br />
<p align="center">
<img src = pics/README_logo.png width= 500px" />
</p>

# SQcircuit: Superconducting Quantum Circuit Analyzer
[**What is SQcircuit?**](#What-is-SQcircuit?)
|[**Installation**](#Installation)
|[**Documentation**](#Documentation)
|[**Examples**](#Examples)
|[**Contribution**](#Contribution)

[![license](https://img.shields.io/badge/license-New%20BSD-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![codecov](https://codecov.io/gh/stanfordLINQS/SQcircuit/branch/main/graph/badge.svg?token=6FT6L9ZPHP)](https://codecov.io/gh/stanfordLINQS/SQcircuit)
## What is SQcircuit?

SQcircuit is an open-source Python library that is capable of analyzing an 
arbitrary superconducting quantum circuit. SQcircuit uses the theory discussed 
in [Rajabzadeh et al., 2022] to describe the Hamiltonian in the appropriate 
basis and to effectively find the energy spectrum and eigenvectors of the 
circuit. To design the desired quantum circuit and to discover new qubits, 
additional functionalities and methods are provided to extract the circuit 
properties such as matrix elements, dephasing rate, decay rates, etc.

Theory detail of SQcircuit and the introduction to library functionalities 
are presented in the following paper:

> Taha Rajabzadeh, Zhaoyou Wang, Nathan Lee, Takuma Makihara, Yudan Guo, 
> Amir H. Safavi-Naeini,<br>
> *Analysis of arbitrary superconducting quantum circuits accompanied by a 
> Python package: SQcircuit*,<br>
> arXiv:2206.08319 (2022),<br>
> https://arxiv.org/abs/2206.08319


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

