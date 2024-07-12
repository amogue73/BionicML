**Syntax of BionicML.py:**

BionicML.py \[-m\] \<GA/PSO/ACO/CS/WOA\> \<number of agents\> \<number of iterations\> \<number of processes\> \[\<desired number of features\>\]
<ul>
  <li>The first argument is optional. If present, the "measure mode" will be used. This means that execution time and energy consumption will be measured and saved in the "Measurements" folder.</li>
  <li>The second argument is the algorithm to be used.</li>
  <li>Arguments 3 to 5 are parameters of the algorithm.</li>
  <li>The last argument is the number of features around which the algorithm will search. It's optional and defaults to 100</li>
</ul>

**Syntax of ACO.py:**

ACO.py \<number of agents\> \<number of iterations\> \<number of processes\>

**Syntax of LUT.py and Linear_regression.py:**

The programs LUT.py and Linear_regression.py don't need arguments.
<ul>
  <li>LUT.py produces the LUT.csv file.</li>
  <li>Linear_regression.py adjust the time and energy formulas using the measurements found in the "Experimental_Measurements" folder</li>
</ul>
