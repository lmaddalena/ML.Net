using System;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;
using AwokeKnowing.GnuplotCSharp;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using MathNet.Numerics.Statistics;

namespace ex2_reg
{
    class Program
    {
        static void Main(string[] args)
        {
            if(!System.Console.IsOutputRedirected)
                System.Console.Clear();

            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

            System.Console.WriteLine("Regularized Logistic Regression ex.2");
            System.Console.WriteLine("====================================\n");

            var M = Matrix<double>.Build;
            var V = Vector<double>.Build;

            // Load Data
            // The first two columns contains the X values and the third column
            // contains the label (y).
            Matrix<double> data = DelimitedReader.Read<double>("data\\ex2data2.txt", false, ",", false);
            Console.WriteLine(data);

            Matrix<double> X = data.SubMatrix(0, data.RowCount, 0, 2);
            Vector<double> y = data.Column(2);
            System.Console.WriteLine("Features:\n");
            System.Console.WriteLine(X);

            System.Console.WriteLine("Label:\n");
            System.Console.WriteLine(y);

            PlotData(X, y);

            Pause();

            //  =========== Part 1: Regularized Logistic Regression ============
            //  In this part, you are given a dataset with data points that are not
            //  linearly separable. However, you would still like to use logistic
            //  regression to classify the data points.
            //
            //  To do so, you introduce more features to use -- in particular, you add
            //  polynomial features to our data matrix (similar to polynomial
            //  regression).
            //

            // Add Polynomial Features

            // Note that mapFeature also adds a column of ones for us, so the intercept
            // term is handled

            X = MapFeature(X.Column(0), X.Column(1));
            System.Console.WriteLine("Mapped features:\n");
            System.Console.WriteLine(X);
            Pause();

            // Initialize fitting parameters
            Vector<double>initial_theta = V.Dense(X.ColumnCount, 0.0);

            // Set regularization parameter lambda to 1
            double lambda = 1;

            // Compute and display initial cost and gradient for regularized logistic
            // regression
            LogisticRegression lr = new LogisticRegression(X, y);
            lr.Lambda = lambda;
            double J = lr.Cost(initial_theta);
            Vector<double> grad = lr.Gradient(initial_theta);

            System.Console.WriteLine("Cost at initial theta (zeros): {0:f5}\n", J);
            System.Console.WriteLine("Expected cost (approx): 0.693\n");
            System.Console.WriteLine("Gradient at initial theta (zeros) - first five values only:\n");
            System.Console.WriteLine(" {0:f5} \n", grad.SubVector(0, 5));
            System.Console.WriteLine("Expected gradients (approx) - first five values only:\n");
            System.Console.WriteLine(" 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n");

            Pause();

            // Compute and display cost and gradient
            // with all-ones theta and lambda = 10
            Vector<double> test_theta = V.Dense(X.ColumnCount, 1.0);
            lr.Lambda = 10;
            J = lr.Cost(test_theta);
            grad = lr.Gradient(test_theta);

            System.Console.WriteLine("\nCost at test theta (with lambda = 10): {0:f5}\n", J);
            System.Console.WriteLine("Expected cost (approx): 3.16\n");
            System.Console.WriteLine("Gradient at test theta - first five values only:\n");
            System.Console.WriteLine(" {0:f5} \n", grad.SubVector(0, 5));
            System.Console.WriteLine("Expected gradients (approx) - first five values only:\n");
            System.Console.WriteLine(" 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n");

            Pause();

            //// ============= Part 2: Regularization and Accuracies =============
            //  Optional Exercise:
            //  In this part, you will get to try different values of lambda and
            //  see how regularization affects the decision coundart
            //
            //  Try the following values of lambda (0, 1, 10, 100).
            //
            //  How does the decision boundary change when you vary lambda? How does
            //  the training set accuracy vary?
            //

            // Initialize fitting parameters
            initial_theta = V.Dense(X.ColumnCount, 0.0);

            // Set regularization parameter lambda to 1 (you should vary this)
            lambda = 1;

            // Optimize
            lr.Lambda = lambda;
            var obj = ObjectiveFunction.Gradient(lr.Cost, lr.Gradient);
            var solver = new BfgsMinimizer(1e-5, 1e-5, 1e-5, 400);            
            var result = solver.FindMinimum(obj, initial_theta);

            // Plot Boundary
            PlotDecisionBoundary(result.MinimizingPoint);
            GnuPlot.HoldOff();

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
            System.Console.WriteLine("Expected accuracy (with lambda = 1): 83.1 (approx)\n");

            Pause();
        }

        private static void PlotDecisionBoundary(Vector<double> theta)
        {
                        // Grid over which we will calculate J
            double[] x1 = MathNet.Numerics.Generate.LinearSpaced(50, -1, 1.5);
            double[] x2 = MathNet.Numerics.Generate.LinearSpaced(50, -1, 1.5);
            
            // initialize J_vals to a matrix of 0's
            int size = x1.Length * x2.Length;
            double[] sx = new double[size];
            double[] sy = new double[size];
            double[] sz = new double[size];

            int idx = 0;
            for(int i = 0; i < x1.Length; i++)
            {
                for(int j = 0; j < x2.Length; j++)
                {
                    sx[idx] = x1[i];
                    sy[idx] = x2[j];

                    Vector<double> v1 = Vector<double>.Build.DenseOfArray(new [] {x1[i]} );
                    Vector<double> v2 = Vector<double>.Build.DenseOfArray(new [] {x2[j]} );
                    Matrix<double> X = MapFeature(v1, v2);
                    Vector<double> z = X * theta; 
                    
                    sz[idx] = z[0];
                    idx++;

                }
            }
            GnuPlot.Set("cntrparam levels discrete 0"); 
            GnuPlot.Contour(sx, sy, sz, "title \"Decision Boundary\"");

        }

        private static Matrix<double> MapFeature(Vector<double> x1, Vector<double> x2)
        {
            // MAPFEATURE Feature mapping function to polynomial features
            //
            //   MAPFEATURE(X1, X2) maps the two input features
            //   to quadratic features used in the regularization exercise.
            //
            //   Returns a new feature array with more features, comprising of 
            //   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
            //
            //   Inputs X1, X2 must be the same size
            //

            Matrix<double> res = Matrix<double>.Build.Dense(x1.Count,1, 1.0);

            int degree = 6;

            for(int i = 1; i <= degree; i++)
            {
                for(int j = 0; j <= i; j++)
                {
                    var v = (x1.PointwisePower(i-j)).PointwiseMultiply(x2.PointwisePower(j)).ToColumnMatrix();
                    res = res.Append(v);
                }
            }

            return res;
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
            GnuPlot.Set("title \"\"");
            GnuPlot.Set("xlabel \"Microchip test 1\"");
            GnuPlot.Set("ylabel \"Microchip test 2\"");
            GnuPlot.Plot(pos.Select(p => p.x1).ToArray(), pos.Select(p => p.x2).ToArray(), "pt 1 ps 1 lc rgb \"black\" title \"y=1\"");
            
            GnuPlot.Plot(neg.Select(p => p.x1).ToArray(), neg.Select(p => p.x2).ToArray(), "pt 7 ps 1 lc rgb \"yellow\" title \"y=0\"");
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
