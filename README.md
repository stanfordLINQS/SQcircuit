<p align="center">
<img src = pics/README_logo.png width= 250px" />
</p>

# SQcircuit: superconducting quantum circuit analyzer
[**What is SQcircuit?**](#What-is-SQcircuit?)
|[**Installation**](#Installation)
|[**Documentation**](#Documentation)
|[**Quick Tutorial**](#Quick-Tutorial)
|[**Examples**](#Examples)
|[**Contribution**](#Contribution)

[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![codecov](https://codecov.io/gh/stanfordLINQS/SQcircuit/branch/main/graph/badge.svg?token=6FT6L9ZPHP)](https://codecov.io/gh/stanfordLINQS/SQcircuit)
## What is SQcircuit?

SQcircuit is an open-source Python library that is capable of analyzing an arbitrary superconducting quantum circuit.
SQcircuit uses the theory discussed in [1] to describe the Hamiltonian in the appropriate basis and to effectively find
the energy spectrum and eigenvectors of the circuit. To design the desired quantum circuit and to discover new qubits, 
additional functionalities and methods are provided to extract the circuit properties such as matrix elements, 
dephasing rate, decay rates, etc.

[1] To be published soon

## Installation
SQcirucit can be simply installed via pip:
```
pip install SQcircuit
```

## Documentation
The documentation of the SQcircuit is provided at: to be published soon

## Quick Tutorial

To show a quick overview of how to use SQcircuit, we find the qubit frequency for the symmetric zero-pi qubit with the
following parameters in GHz: E_C=0.15, E_CJ=10, E_L=0.13 , and E_J=5.   

<p align="center">
<img src = pics/README_zeroPi.png width= "280px" />
</p>
After installing the SQcircuit, we import it via:

```python
# import the SQcircuit library
import SQcircuit as sq
```
Since zero-pi qubit has a single inductive loop, we define its loop by creating a loop object from `Loop` class with
flux bias at frustration point:

```python
# inductive loop of zero-pi qubit with flux bias at its frustration point.
loop1 = sq.Loop(value=0.5)
```
We can later change the value of the flux bias by `setFlux()` method. Each circuit component in SQcircuit has their
own class definition `Capacitor` class for capacitors, `Inductor` class for inductors, and `Junction` class for
Josephson junctions. We define the elements of our zero-pi circuit as following:
```python
# capacitors
C = sq.Capacitor(value =0.15 ,  unit="GHz")
CJ = sq.Capacitor(value=10, unit="GHz")
# inductors
L = sq.Inductor(value=0.13, unit="GHz", loops = [loop1])
# JJs
JJ = sq.Junction(value=5, unit="GHz", loops=[loop1])
```
Note that for the inductive elements( inductors as well as Josephson junctions) that are part of an 
inductive loop, one should indicate the loops of which they are involved. For example here we pass `[loop1]` to `loops`
argument for both inductors and Josephson Junctions, because all of them are part of `loop1`. After defining all
components of the circuit, to describe the circuit topology in SQcircuit, one should create an object of `Circuit`
class by passing a Python dictionary that contains the list of all elements at each edge

```python
# dictionary that contains the list of all elements at each edge
elements = {(0, 1): [CJ, JJ],
            (0, 2): [L],
            (0, 3): [C],
            (1, 2): [C],
            (1, 3): [L],
            (2, 3): [CJ, JJ]}

# define the circuit
cr = sq.Circuit(elements)
```
One step before diagonalizing the circuit is to define the size of the Hilbert space by specifying the truncation 
numbers for each mode.(For more information about modes and truncation number check the SQcircuit original paper or
the documentation)

```python
# call this function to set the truncation number for each mode of the circuit. 
cr.truncationNumbers([25, 1, 25])
```
We get the first two eigenfrequencies of the circuit to calculate the qubit frequency via:

```python
# get the first two eigenfrequencies and eigenvectors 
eigFreq, eigVec = cr.diag(numEig=2)

# print the qubit frequency
print("qubit frequency:", eigFreq[1]-eigFreq[0])
```
The frequency unit in SQcircuit is GHz by default. However, one can simply change it by `sq.unit.setFreq()` method.

## Examples

To manifest the potential of the SQcircuit, we prepared the examples from simple qubits to state of the art 
super conducting circuits of the literature, in which we effortlessly reproduce the main result of the paper 
by SQcircuit functionalities. One can find the jupyter notebook examples in example folder, some of which are:

* [Zero-Pi Qubit](https://github.com/stanfordLINQS/SQcircuit/blob/main/examples/zeroPiQubit.ipynb): We calculated the
energy spectrum and eigenfunctions of the zero-pi qubit in the 
[Groszkowski2018](https://iopscience-iop-org.stanford.idm.oclc.org/article/10.1088/1367-2630/aab7cd)
* [Inductively Shunted Circuit](https://github.com/stanfordLINQS/SQcircuit/blob/main/examples/inductivelyShunted.ipynb):
[Smith2016](https://journals-aps-org.stanford.idm.oclc.org/prb/abstract/10.1103/PhysRevB.94.144507)
explained how the conventional method or perturbation theory does not correctly diagonalize their 
highly anharmonic inductively-shunted qubits. However, by using SQcircuit, we simply reproduced the energy spectrum.
* [Qubit protected by two Cooper-pair tunneling](https://github.com/stanfordLINQS/SQcircuit/blob/main/examples/twoCPB.ipynb):
[Smith2020](https://doi-org.stanford.idm.oclc.org/10.1038/s41534-019-0231-2)
designed a qubit that is protected by two Cooper-pair tunneling. We reproduced the main results of the paper such as
energy spectrum, wavefunctions, and matrix elements by use of SQcircuit.

## Contribution
You are very welcome to contribute to SQcircuit development by forking this repository and sending pull requests,
or filing bug reports at the [issues](https://github.com/stanfordLINQS/SQcircuit/issues) page. All code contributions are acknowledged in the contributors section
in the documentation.

