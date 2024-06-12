import os
import sys

from setuptools import setup, find_packages

DESCRIPTION = "superconducting quantum circuit analyzer"

LONG_DESCRIPTION = """SQcircuit is an open-source Python library that is 
capable of analyzing an arbitrary superconducting quantum circuit. SQcircuit 
uses the theory discussed in [Rajabzadeh et al., 2022] to describe the 
Hamiltonian in the appropriate basis and to effectively find the energy 
spectrum and eigenvectors of the circuit. To design the desired quantum 
circuit and to discover new qubits, additional functionalities and methods 
are provided to extract the circuit properties such as matrix elements, 
dephasing rate, decay rates, etc.

Theory detail of SQcircuit and the introduction to library functionalities are 
presented in the following paper:

Taha Rajabzadeh, Zhaoyou Wang, Nathan Lee, Takuma Makihara, Yudan Guo, 
Amir H. Safavi-Naeini, 'Analysis of arbitrary superconducting quantum circuits 
accompanied by a Python package: SQcircuit', arXiv:2206.08319 (2022). 
https://arxiv.org/abs/2206.08319
"""

# version of the SQcircuit
MAJOR = 0
MINOR = 0
PATCH = 15

VERSION = "%d.%d.%d" % (MAJOR, MINOR, PATCH)

CURRENT_DIR = os.path.abspath('.')

with open(os.path.join(CURRENT_DIR, "requirements.txt")) as requirements:
    INSTALL_REQUIRES = requirements.read().splitlines()

setup(
    name="SQcircuit",
    version=VERSION,
    author="Taha Rajabzadeh, Amir Safavi-Naeini",
    author_email="tahar@stanford.edu, safavi@stanford.edu",
    license="BSD",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    keywords="superconducting circuits",
    url="https://github.com/stanfordLINQS/SQcircuit",
    install_requires=INSTALL_REQUIRES,
    zip_safe=False,
    include_package_data=True,
    packages=["SQcircuit", "SQcircuit/tests"],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ]
)
