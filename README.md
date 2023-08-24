# snf
An adaptation of Similarity Network Fusion (SNF) to Python

# Motivation
While an implementation of SNF in Python already exists (https://github.com/rmarkello/snfpy),
I found that the output from their main SNF function differed from the output I got
from using the SNF package developed by the original authors in R (see tests/test_snf.py
in this repository).

`snfpy` is a great library in its own right (some of its functions are
used here as a dependency). I built this small package as an adjunct to provide an
SNF function that is an exact replica of the widely used SNFtools package in R.

This pacakge also includes an implementation of the `robust_core_clustering`
function from [Jacobs et al. (2021)](https://www.nature.com/articles/s41386-020-00902-6).

# Getting started
After cloning this repository locally, you can install it by navigating to the
main package folder and running the command
```
pip install .
```

To import the main SNF function, you can use this line of Python code:
```
from snftools.snf import SNF
```
