<div align="center">
  <img alt="Logo image" src="https://raw.githubusercontent.com/stanfordLINQS/SQcircuit/main/pics/light_logo_sq.png" width="150" height="auto">
</div>

# SQcircuit

SQcircuit is an open-source Python library that is capable of analyzing an arbitrary superconducting quantum circuit. SQcircuit uses the theory discussed in [Rajabzadeh et al., 2022] to describe the Hamiltonian in the appropriate basis and to effectively find the energy  spectrum and eigenvectors of the circuit. To design the desired quantum circuit and to discover new qubits, additional functionalities and methods are provided to extract the circuit properties such as matrix elements, dephasing rate, decay rates, etc.

Details about the theory behind the SQcircuit core code and an introduction to the library's functionalities are provided in the following paper:

> Taha Rajabzadeh, Zhaoyou Wang, Nathan Lee, Takuma Makihara, Yudan Guo, Amir H. Safavi-Naeini, "Analysis of arbitrary superconducting quantum circuits  accompanied by a Python package: SQcircuit", Quantum 7, 1118, https://quantum-journal.org/papers/q-2023-09-25-1118/.

With the v1.0 release, SQcircuit can also compute gradients of arbitrary circuit properties, using PyTorch.

# TODO: finish writing
