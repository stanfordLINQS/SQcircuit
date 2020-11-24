# Qcircuit
[**What is Qcircuit?**](#What-is-Qcircuit?)
|[**Quick Tutorial**](#Quick-Tutorial)
|[**Installation**](#Installation)
|[**Examples**](#Examples)

## What is Qcircuit?

Qcircuit is a superconducting quantum circuit solver that is written as Python class. By giving the circuit configuration and its parameter as an input to this solver, it is capable of generating the Hamiltonian for the circuit and find the eigenvalues and eigenfunctions of the Hamiltonian effectively.


## Quick Tutorial

This tutorial shows an overview of how to use Qcircuit. For more details, one can visit the example file which contains a variety examples from the state of the art literature on superconducting quantum circuits.

<p align="center">
<img src = pics/README_Pic1.png width= "550px" />
</p>

First step is to create an object of the Qcircuit class by the following code:

```python
# Import circuitClass that contains Qcircuit
from circuitClass import *

# cicuitParam is a dictionary that contains the information about the graph structure,
# capacitor values, inductor values, and Josephson Junction Values.
circuitParam = {
	'graph':[[0,1],[1,2],[2,3],[0,2],[0,3]],
	'capacitors':[C1,C2,C3,C4,C5],
	'inductors':[L1,None,None,L2,None],
	'JJs':[[EJ1,EJ2],EJ3,None,None,EJ4]
}

# cr is an object of Qcircuit
cr = Qcircuit(circuitParam)
```

We create the object of the Qcircuit by initializing it with `circuitParam` dictionary. The `circuitParam['graph']` is a list of edges that specifies the graph structure of the circuit. The rest of parameters `circuitParam['capacitors']`, `circuitParam['inductors']`, and `circuitParam['JJs']` are list of capacitor values, inductor values, Josephson Junction Values in the same order of their edges in the `circuitParam['graph']`. For the elements that are parallel with each other in one edge we put them as list such as `[EJ1,EJ2]` for the edge `[0,1]`.

The second step is to set the accuracy of the solver and to set up the equations needed to calculate the Hamiltonian of the circuit: 

```python
# call this function to set the truncation number for each mode of the circuit. 
cr.setModeNumbers([m1,m2,m3])

# call this function to set up equations and preprocesses needed to calculate the Hamiltonian.
cr.configure()
```

`cr.setModeNumbers([m1,m2,m3])` sets the truncation number for each modes of the circuit. Each element of `[m1,m2,m3]` should be an integer, and they should be large enough that Qcircuit converges. By calling `cr.configure()` Qcircuit sets up the equations needed to calculate the Hamiltonian. Additionally, this function preprocess and calculate the quantum operators and part of the Hamiltonian that are independent from external fluxes.

Next, we find the inductive loops that we can apply external fluxes to them.

```python
cr.getExternalLinks()
# Qcircuit returns [[0,1,2],[0,1],[0,1]], which means that we can apply external fluxes to loop created 
# by the [0,1,2] cycle and two loops created at the edge [0,1]
```


we call `cr.getExternalLinks()` to see how Qcircuit sees the inductive loops that we can apply external fluxes to that cycles. In this circuit, Qcircuit returns the `[[0,1,2],[0,1],[0,1]]`, which means that we can apply external fluxes to loop created by the `[0,1,2]` cycle and two loops created at the edge `[0,1]`.

After that we apply external fluxes to inductive loops and run the solver to find the eigenvalues and eigenvectors of the Hamiltonian corresponds to the input circuit. 

```python
# set external fluxes for each inductive loops of the circuit.
cr.setExcitation([phi1,phi2,phi3])

# run the solver to calculate the eigenvalues and eigenvectors of the Hamiltonian for 
# specific number of bands
cr.run(numBand = N)
```

`cr.setExcitation([phi1,phi2,phi3])` set external fluxes for each inductive loops of the circuit. One should apply `[phi1,phi2,phi3]` list with the same order as `[[0,1,2],[0,1],[0,1]]` that `cr.getExternalLinks()` gave us. `cr.run(numBand = N)` calculate the first `N` eigenvalues and eigenvectors of the circuit.  

The final results are stored in the two following variables:

```python
# the array that contains the eigenvalues calculated by Qcircuit  
cr.HamilEigVal

# the list that contains the eigenvectors calculated by Qcircuit 
cr.HamilEigVecList
```

## Installation
To use Qcircuit, you just need to put `circuitClass.py` and `PhysicsConstants.py` from [source](https://github.com/taha1373/Qcircuit/tree/master/source) folder inside your project folder and follow the [**Quick Tutorial**](#Quick-Tutorial) and [**Examples**](#Examples) sections.

## Examples

To see how Qcircuit is working and its robustness, we put several examples from the sate of the art superconducting circuits in the litreture, which Qcircuit can efficiently calculate the spectrum of those circuits.

* [Zero-Pi Qubit](https://github.com/taha1373/Qcircuit/blob/master/examples/zeroPiQubit.ipynb)
* [Inductively Shunted Circuits](https://github.com/taha1373/Qcircuit/blob/master/examples/inductivelyShunted.ipynb)
* [4 Local Coupler Circuit](https://github.com/taha1373/Qcircuit/blob/master/examples/4localCoupler.ipynb)

