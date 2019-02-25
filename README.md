ML.Net
========
Coursera's Machine Leaarning course assignement implemented in DotNet Core using Visual Studio Code.

Numerical computations are made using the MathNet libraries (https://www.mathdotnet.com/). 
For plotting I'm using GnuPlot (http://www.gnuplot.info/) interfaced with the GnuplotCSharp library implemented by James Morris aka AwokeKnowing (https://github.com/AwokeKnowing/GnuplotCSharp)

ex1
--------
**Linear Regresion**

```
Cost and Gradient descent....

With theta = [0 ; 0]
Cost computed = 32.07

Expected cost value (approx) 32.07

With theta = [-1 ; 2]
Cost computed = 54.24

Expected cost value (approx) 54.24

Running Gradient Descent ...

Theta found by gradient descent:

DenseMatrix 2x1-Double
-3.63029
 1.16636

Expected theta values (approx)

 -3.6303
  1.1664

For population = 35,000, we predict a profit of 4519.7679
For population = 70,000, we predict a profit of 45342.4501

```

Plot of data and linear fit

![Linear Regression](https://github.com/lmaddalena/ML.Net/tree/master/ex1/images/LinearRegression.png)


**Cost function J**
Surface plot of cost function J

![Cost function J](https://github.com/lmaddalena/ML.Net/tree/master/ex1/images/CostJ.png)

**Contour plot of cost function J**
Countour plot of cost function J

![Cost function J](https://github.com/lmaddalena/ML.Net/tree/master/ex1/images/contour.png)
