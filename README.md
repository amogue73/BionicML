# BionicML

This program allows to execute five types of parallel bioinspired algorithms for Feature Selection (FS). These algorithms work with a dataset of Electroencephalography (EGG) and consist of wrapper methods which use the K-nearest neighbors (KNN) algorithm to calculate the fitness of the subsets of features. The output of the program is the best subset of features found as well as the accuracy achieved by KNN.

The program has an additional option to measure the execution time and energy consumed by the processor during execution. These data, as well as the solution and accuracy are registered in the folder of the respective algorithm used. The purpose of this feature is to adjust formulas of time and energy by linear regression, which is done by the program Linear_regression.py

# Version

1.0

# Author

Alejandro Moreno Guerrero (amogue@correo.ugr.es) and Juan José Escobar Pérez (jjescobar@ugr.es)

# Requirements

Python 3

# Usage

The syntax to use the program is described in "User guide.md"

# Funding

This work has been funded by:

<ul>
  <li>Spanish Ministerio de Ciencia, Innovación y Universidades under grant numbers PID2022-137461NB-C32 and PID2020-119478GB-I00.</li>
  <li>European Regional Development Fund (ERDF).</li>
  <li>Universidad de Granada, under grant number PPJIA2023-025.</li>
</ul>

# Copyright

BionicML © [Universidad de Granada](https://www.ugr.es/)

