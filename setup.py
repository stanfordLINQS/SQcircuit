import os
import sys

from setuptools import setup, find_packages

DOCLINES = __doc__.split("\n")

# version of the SQcircuit
MAJOR = 0
MINOR = 0
MICRO = 1
ISRELEASED = True

VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CURRENT_DIR, "requirements.txt")) as requirements:
    INSTALL_REQUIRES = requirements.read().splitlines()


EXTRAS_REQUIRE = {
    "graphics": ["matplotlib-label-lines (>=0.3.6)"],
    "explorer": ["ipywidgets (>=7.5)"],
    "h5-support": ["h5py (>=2.10)"],
    "pathos": ["pathos", "dill"],
    "fitting": ["lmfit"],
}


DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
CLASSIFIERS = [_f for _f in CLASSIFIERS.split("\n") if _f]


def git_short_hash():
    try:
        git_str = "+" + os.popen('git log -1 --format="%h"').read().strip()
    except OSError:
        git_str = ""
    else:
        if git_str == "+":  # fixes setuptools PEP issues with versioning
            git_str = ""
    return git_str


FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += ".dev" + str(MICRO) + git_short_hash()


def write_version_py(filename="SQcircuit/version.py"):
    cnt = """\
# THIS FILE IS GENERATED FROM SQcircuit SETUP.PY
short_version = '%(version)s'
version = '%(fullversion)s'
release = %(isrelease)s
"""
    versionfile = open(filename, "w")
    try:
        versionfile.write(
            cnt
            % {
                "version": VERSION,
                "fullversion": FULLVERSION,
                "isrelease": str(ISRELEASED),
            }
        )
    finally:
        versionfile.close()


local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(local_path)
sys.path.insert(0, local_path)
sys.path.insert(0, os.path.join(local_path, "SQcircuit"))  # to retrieve _version

# always rewrite _version
if os.path.exists("SQcircuit/version.py"):
    os.remove("SQcircuit/version.py")
write_version_py()

setup(
    name="SQcircuit",
    version=FULLVERSION,
    author="Taha Rajabzadeh, Amir Safavi-Naeini",
    author_email="tahar@stanford.edu, safavi@stanford.edu",
    license="BSD",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    keywords="superconducting circuits",
    url="https://github.com/stanfordLINQS/SQcircuit",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    zip_safe=False,
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
