# BionicML

This program allows to execute five types of parallel bioinspired algorithms for Feature Selection (FS). These algorithms work with a dataset of Electroencephalography (EGG) and consist of wrapper methods which use the K-nearest neighbors (KNN) algorithm to calculate the fitness of the subsets of features. The output of the program is the best subset of features found as well as the accuracy achieved by KNN.

The program has an additional option to measure the execution time and energy consumed by the processor during execution. These data, as well as the solution and accuracy are registered in the folder of the respective algorithm used. The purpose of this feature is to adjust formulas of time and energy by linear regression, which is done by the program Linear_regression.py

# Usage

The syntax of BionicML.py is the following:
BionicML.py \[-m\] \<GA/PSO/ACO/CS/WOA\> \<number of agents\> \<number of iterations\> \<number of processes\> \[\<desired number of features\>\]
<ol>
  <li>The first argument is optional. If present, the "measure mode" will be used. This means that execution time and energy consumption will be measured and saved in the "Measurements" folder.</li>
  <li>The second argument is the algorithm to be used.</li>
  <li>Arguments 3 to 5 are parameters of the algorithm.</li>
  <li>The last argument is the number of features around which the algorithm will search.</li>
</ol>


The programs LUT.py and Linear_regression.py don't need arguments.
<ol>
  <li>LUT.py produces the LUT.csv file.</li>
  <li>Linear_regression.py adjust the time and energy formulas using the measurements found in the "Experimental_Measurements" folder</li>
</ol>


