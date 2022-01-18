# SQcircuit
[**What is SQcircuit?**](#What-is-SQcircuit?)
|[**Quick Tutorial**](#Quick-Tutorial)
|[**Installation**](#Installation)
|[**Examples**](#Examples)

## What is SQcircuit?

SQcircuit is a superconducting quantum circuit solver that is written as Python class. By giving the circuit configuration and its parameter as an input to this solver, it is capable of generating the Hamiltonian for the circuit and find the eigenvalues and eigenfunctions of the Hamiltonian effectively.


## Quick Tutorial

This tutorial shows an overview of how to use SQcircuit. For more details, one can visit the example file which contains a variety examples from the state of the art literature on superconducting quantum circuits.

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


## Installation
To use SQcircuit, you just need to put `circuit.py` and `PhysicsConstants.py` from [source](https://github.com/taha1373/SQcircuit/tree/master/source) folder inside your project folder and follow the [**Quick Tutorial**](#Quick-Tutorial) and [**Examples**](#Examples) sections.

## Examples

To see how SQcircuit is working and its robustness, we put several examples from the sate of the art superconducting circuits in the litreture, which SQcircuit can efficiently calculate the spectrum of those circuits.

* [Zero-Pi Qubit](https://github.com/stanfordLINQS/Qcircuit/blob/main/examples/zeroPiQubit.ipynb)
* [Inductively Shunted Circuits](https://github.com/stanfordLINQS/Qcircuit/blob/main/examples/inductivelyShunted.ipynb)


