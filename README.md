# <span style="color:Orange">*SQ*</span>circuit: superconducting quantum circuit analyzer
[**What is SQcircuit?**](#What-is-SQcircuit?)
|[**Installation**](#Installation)
|[**Documentation**](#Documentation)
|[**Quick Tutorial**](#Quick-Tutorial)
|[**Examples**](#Examples)

## What is SQcircuit?

SQcircuit is an open-source Python library that is capable of analyzing an arbitrary superconducting quantum circuit.
SQcircuit uses the theory discussed in [1] to describe the Hamiltonian in the appropriate basis and to effectively find
the energy spectrum and eigenvectors of the circuit. To design the desired quantum circuit and to discover new qubits, 
additional functionalities and methods are provided to extract the circuit properties such as matrix elements, 
dephasing rate, decay rates, etc.

## Installation
SQcirucit can be simply installed via pip:
```
pip install SQcircuit
```
As an alternative, installation via Conda is also provided.
```
conda install -c conda-forge SQcircuit
```
## Documentation
THe documentation of the SQcircuit is provided at:

## Quick Tutorial

This tutorial shows an overview of how to use SQcircuit. For more details, one can visit the example file which
contains a variety examples from the state of the art literature on superconducting quantum circuits.

<p align="center">
<img src = pics/README_Pic1.png width= "550px" />
</p>


```python
# Import circuitClass that contains SQcircuit
from circuit import *

# cicuitParam is a dictionary that contains the information about the graph structure,
# capacitor values, inductor values, and Josephson Junction Values.
circuitParam = {
    (0, 1): {"C": C1, "L": L1, "JJ": [EJ1, EJ2]},
    (1, 2): {"C": C2, "JJ": EJ3},
    (2, 3): {"C": C3},
    (0, 2): {"C": C4, "L": L2},
    (0, 3): {"C": C5, "JJ": EJ4}
}

# cr is an object of SQcircuit
cr = SQcircuit(circuitParam)
```


```python
# call this function to set the truncation number for each mode of the circuit. 
cr.setTruncationNumbers([m1,m2,m3])
```

```python
# set external fluxes for each inductive loops of the circuit.
cr.setExternalFluxes({(1, 2): phi1,
                      (0, 1): [phi2,phi3]
                      })

# run the solver to calculate the eigenvalues and eigenvectors of the Hamiltonian for 
# specific number of bands
eigenValues, eigenVectros = cr.run(numBand = N)
```

## Examples

To see how SQcircuit is working and its robustness, we put several examples from the sate of the art
superconducting circuits in the litreture, which SQcircuit can efficiently calculate the spectrum of those circuits.

* [Zero-Pi Qubit](https://github.com/stanfordLINQS/Qcircuit/blob/main/examples/zeroPiQubit.ipynb)
* [Inductively Shunted Circuits](https://github.com/stanfordLINQS/Qcircuit/blob/main/examples/inductivelyShunted.ipynb)


