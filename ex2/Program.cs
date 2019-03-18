using System;
using System.Linq;
using System.Globalization;
using AwokeKnowing.GnuplotCSharp;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using System.Collections.Generic;

namespace ex2
{
    class Program
    {
        static void Main(string[] args)
        {
            if(!System.Console.IsOutputRedirected)
                System.Console.Clear();

            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

            System.Console.WriteLine("Logistic Regression ex.2");
            System.Console.WriteLine("========================\n");

            var M = Matrix<double>.Build;
            var V = Vector<double>.Build;

            // Load Data
            // The first two columns contains the exam scores and the third column
            // contains the label.
            Matrix<double> data = DelimitedReader.Read<double>("data\\ex2data1.txt", false, ",", false);
            Console.WriteLine(data);

            Matrix<double> X = data.SubMatrix(0, data.RowCount, 0, 2);
            Vector<double> y = data.Column(2);
            System.Console.WriteLine("Features:\n");
            System.Console.WriteLine(X);

            System.Console.WriteLine("Label:\n");
            System.Console.WriteLine(y);

            // ==================== Part 1: Plotting ====================
            //  We start the exercise by first plotting the data to understand the 
            //  the problem we are working with.

            System.Console.WriteLine("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n");
            PlotData(X, y);
            GnuPlot.HoldOff();

            Pause();

            // theta parameters
            Vector<double> initial_theta = V.Dense(X.ColumnCount + 1);

            // Add intercept term to X
            X = X.InsertColumn(0, V.Dense(X.RowCount, 1));

            // compute cost
            LogisticRegression lr = new LogisticRegression(X, y);
            double J = lr.Cost(initial_theta);
            Vector<double> grad = lr.Gradient(initial_theta);
            System.Console.WriteLine("Cost at initial theta (zeros): {0:f3}\n", J);
            System.Console.WriteLine("Expected cost (approx): 0.693\n");

            System.Console.WriteLine("Gradient at initial theta (zeros): \n");
            System.Console.WriteLine(" {0:f4} \n", grad);
            System.Console.WriteLine("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n");

            // Compute and display cost and gradient with non-zero theta
            Vector<double> test_theta = V.DenseOfArray(new double[] {-24.0, 0.2, 0.2});
            J = lr.Cost(test_theta);
            grad = lr.Gradient(test_theta);

            System.Console.WriteLine("\nCost at test theta: {0:f3}\n", J);
            System.Console.WriteLine("Expected cost (approx): 0.218\n");
            System.Console.WriteLine("Gradient at test theta: \n");
            System.Console.WriteLine(" {0:f3} \n", grad);
            System.Console.WriteLine("Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n");
            
            Pause();

            // ============= Part 3: Optimizing using fmin  ================
            //  In this exercise, I will use fmin function  to find the
            //  optimal parameters theta.
            var obj = ObjectiveFunction.Gradient(lr.Cost, lr.Gradient);
            var solver = new BfgsMinimizer(1e-5, 1e-5, 1e-5, 1000);            
            var result = solver.FindMinimum(obj, initial_theta);

            System.Console.WriteLine("Cost at theta found by fmin: {0:f5} after {1} iterations\n", result.FunctionInfoAtMinimum.Value, result.Iterations);
            System.Console.WriteLine("Expected cost (approx): 0.203\n");
            System.Console.WriteLine("theta: \n");
            System.Console.WriteLine(result.MinimizingPoint);
            System.Console.WriteLine("Expected theta (approx):\n");
            System.Console.WriteLine(" -25.161\n 0.206\n 0.201\n");

            Pause();

            PlotLinearBoundary(X, y, result.MinimizingPoint);
            GnuPlot.HoldOff();

            // ============== Part 4: Predict and Accuracies ==============
            //  After learning the parameters, you'll like to use it to predict the outcomes
            //  on unseen data. In this part, you will use the logistic regression model
            //  to predict the probability that a student with score 45 on exam 1 and 
            //  score 85 on exam 2 will be admitted.
            //
            //  Furthermore, you will compute the training and test set accuracies of 
            //  our model.
            //
            //  Your task is to complete the code in predict.m

            //  Predict probability for a student with score 45 on exam 1 
            //  and score 85 on exam 2 

            double prob = LogisticRegression.Predict(V.DenseOfArray(new [] {1.0, 45.0, 85.0}), result.MinimizingPoint);
            System.Console.WriteLine("For a student with scores 45 and 85, we predict an admission probability of {0:f5}\n", prob);
            System.Console.WriteLine("Expected value: 0.775 +/- 0.002\n\n");

            Pause();

            // Compute accuracy on our training set
            Vector<double> pos = LogisticRegression.Predict(X, result.MinimizingPoint);            
            Func<double, double> map = delegate(double d){
                if(d >= 0.5)
                    return 1;
                else 
                    return 0;
            };

            pos = pos.Map(map);
            Vector<double> comp = V.Dense(y.Count);
            
            for(int i = 0; i < y.Count; i++)
            {
                if(pos[i] == y[i])
                    comp[i] = 1;
                else
                    comp[i] = 0;
            }

            double accurancy = comp.Mean() * 100;

            System.Console.WriteLine("Train Accuracy: {0:f5}\n", accurancy);
            System.Console.WriteLine("Expected accuracy (approx): 89.0\n");
            System.Console.WriteLine("\n");

        }

        private static void PlotLinearBoundary(Matrix<double> X, Vector<double> y, Vector<double> theta)
        {
                GnuPlot.Set("terminal wxt 1");

                PlotData(X.RemoveColumn(0), y);

                // Only need 2 points to define a line, so choose two endpoints
                double x1 = X.Column(2).Minimum() - 2;
                double x2 = X.Column(2).Maximum() + 2;
                Vector<double> X_plot = Vector<double>.Build.DenseOfArray(new double []{x1, x2});                
                var y_plot = (-1/theta[2]) * (theta[1] * X_plot + theta[0]);
                
                GnuPlot.Plot(X_plot.ToArray(), y_plot.ToArray(), "with lines linestyle 1 linewidth 1 title \"Decision Boundary\" ");
                // GnuPlot.Set("xrange [30:100]");
                // GnuPlot.Set("yrange [30:100]");
        }

        private static void PlotJ(Vector<double> j_history, string title, string color)
        {
            double[] x = MathNet.Numerics.Generate.LinearRange(1,1,j_history.Count);
            double[] y = j_history.ToArray();

            GnuPlot.Set("terminal wxt 1");
            GnuPlot.Set("title \"Cost function J\"");
            GnuPlot.Set("xlabel \"Number of iteration\"");
            GnuPlot.Set("ylabel \"Cost J\"");
            GnuPlot.Plot(x, y, "with lines linestyle 1 lc rgb \""  + color + "\" linewidth 2 title \"" + title + "\" ");
 
        }

        public static void PlotData(Matrix<double> X, Vector<double> y)
        {

            List<(double x1,double x2)> pos = new List<(double x1, double x2)>();
            List<(double x1,double x2)> neg = new List<(double x1, double x2)>();

            for(int i = 0; i<y.Count; i++)
            {
                if(y[i] == 1)
                    pos.Add((X[i,0], X[i,1]));
                else
                    neg.Add((X[i,0], X[i,1]));
            }

                        
            GnuPlot.HoldOn();
            GnuPlot.Set("title \"Data\"");
            //GnuPlot.Set("key bottom right");
            GnuPlot.Set("key outside noautotitle");
            //GnuPlot.Set("key title \"Legend\"");
            GnuPlot.Set("xlabel \"Exam 1 score\"");
            GnuPlot.Set("ylabel \"Exam 2 score\"");
            GnuPlot.Plot(pos.Select(p => p.x1).ToArray(), pos.Select(p => p.x2).ToArray(), "pt 1 ps 1 lc rgb \"black\" title \"Admitted\"");
            
            GnuPlot.Plot(neg.Select(p => p.x1).ToArray(), neg.Select(p => p.x2).ToArray(), "pt 7 ps 1 lc rgb \"yellow\" title \"Not admitted\"");
        }

        // unvectorized versione of cost function
        private static (double cost, Matrix<double> grad) ComputeCostNonVect(Matrix<double> X, Matrix<double> y, Matrix<double> theta)
        {            
            double J = 0;           // cost
            double m = X.RowCount;  // number of examples
            int n = X.ColumnCount;  // number of features

            for(int i = 0; i < m; i++)
            {
                double z = 0;

                for(int j = 0; j < n; j++)
                {
                    z += X[i,j]*theta[j,0];         // X*theta
                }

                double h = 1 / (1 + Math.Exp(-z));  // sigmoid

                J += y[i,0] * Math.Log(h) + (1 - y[i,0]) * Math.Log(1 - h);
            }

            J = - 1/m * J;

            // gradient computation
            Matrix<double> grad = Matrix<double>.Build.Dense(theta.RowCount, theta.ColumnCount);
            for(int j = 0; j < n; j++)
            {
                double g = 0;

                for(int i = 0; i < m; i++)
                {
                    double z = 0;

                    for(int k = 0; k < n; k++)
                    {
                        z += X[i,k] * theta[k,0];       // X*theta
                    }

                    double h = 1 / (1 + Math.Exp(-z));  // sigmoid
                    
                    g += (h - y[i,0]) * X[i,j];
                }

                g = 1/m * g;
                grad[j, 0] = g;
            }



            return (J, grad);
        }

        private static void Pause()
        {
            if(!System.Console.IsOutputRedirected)
            {
                Console.WriteLine("Program paused. Press enter to continue.\n");
                Console.ReadKey();
            }
        }

    }
}
