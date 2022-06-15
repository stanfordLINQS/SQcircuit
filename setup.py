import os
import sys

from setuptools import setup, find_packages

DESCRIPTION = "superconducting quantum circuit analyzer"

# version of the SQcircuit
MAJOR = 0
MINOR = 0
PATCH = 9
ISRELEASED = True

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
