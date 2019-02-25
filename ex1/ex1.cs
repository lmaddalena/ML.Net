using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;
using System.Globalization;
using AwokeKnowing.GnuplotCSharp;

namespace ex1
{
    class Program
    {
        static void Main(string[] args)
        {
            if(!System.Console.IsOutputRedirected)
                System.Console.Clear();

            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

            System.Console.WriteLine("Linear Regression ex.1");
            System.Console.WriteLine("======================\n");

            // ==================== Part 1: Basic Function ====================
            var M = Matrix<double>.Build;
            var V = Vector<double>.Build;

            // load data
            Matrix<double> data = DelimitedReader.Read<double>("data\\ex1data1.txt", false, ",", false);
            Console.WriteLine(data);
 
            Matrix<double> X = data.Column(0).ToColumnMatrix();
            Matrix<double> y = data.Column(1).ToColumnMatrix();
            int m = X.RowCount;

            // ======================= Part 2: Plotting =======================
            System.Console.WriteLine("Plotting data....");
            GnuPlot.HoldOn();
            PlotData(X.Column(0).ToArray(), y.Column(0).ToArray());
            Pause();

            // =================== Part 3: Cost and Gradient descent ===================
            System.Console.WriteLine("Cost and Gradient descent....");

            // Add a column of ones to X
            X = X.InsertColumn(0, V.Dense(m, 1));
            System.Console.WriteLine(X);

            // initialize fitting parameters
            Matrix<double> theta = M.Dense(2, 1);

            double J = ComputeCost(X, y, theta);
            System.Console.WriteLine("With theta = [0 ; 0]\nCost computed = {0:f}\n", J);
            System.Console.WriteLine("Expected cost value (approx) 32.07\n");

            // initialize fitting parameters
            theta[0,0] = -1;  theta[1,0] = 2;
            J = ComputeCost(X, y, theta);
            System.Console.WriteLine("With theta = [-1 ; 2]\nCost computed = {0:f}\n", J);
            System.Console.WriteLine("Expected cost value (approx) 54.24\n");

            // run gradient descent
            System.Console.WriteLine("\nRunning Gradient Descent ...\n");

            int iterations = 1500;
            double alpha = 0.01;
            theta = M.Dense(2, 1);
            (Matrix<double> theta, Matrix<double> J_history) res;
            res = GradientDescent(X, y, theta, alpha, iterations);
            theta = res.theta;

            // print theta to screen
            System.Console.WriteLine("Theta found by gradient descent:\n");
            System.Console.WriteLine(theta);
            System.Console.WriteLine("Expected theta values (approx)\n");
            System.Console.WriteLine(" -3.6303\n  1.1664\n\n");
            
            Matrix<double> h = X*theta;             // hypothesys
            Matrix<double> x = X.RemoveColumn(0);   // remove x0
            PlotLinearFit(x.Column(0).ToArray(), h.Column(0).ToArray());
            GnuPlot.HoldOff();

            var predict1 = M.DenseOfArray(new double[,] {{1, 3.5}}) * theta;
            System.Console.WriteLine("For population = 35,000, we predict a profit of {0:F4}", predict1[0,0]*10000);

            var predict2 = M.DenseOfArray(new double[,] {{1, 7}}) * theta;
            System.Console.WriteLine("For population = 70,000, we predict a profit of {0:F4}", predict2[0,0]*10000);

            Pause();

            // ============= Part 4: Visualizing J(theta_0, theta_1) =============
            System.Console.WriteLine("Visualizing J(theta_0, theta_1) ...\n");
            PlotJ(X, y, theta);

            Pause();
        }


        // Plot J cost function
        private static void PlotJ(Matrix<double> X, Matrix<double> y, Matrix<double> theta)
        {
            // Grid over which we will calculate J
            double[] theta0_vals = MathNet.Numerics.Generate.LinearSpaced(100,-10,10);
            double[] theta1_vals = MathNet.Numerics.Generate.LinearSpaced(100,-1,4);

            // initialize J_vals to a matrix of 0's
            int size = theta0_vals.Length * theta1_vals.Length;
            double[] sx = new double[size];
            double[] sy = new double[size];
            double[] sz = new double[size];


            // Fill out J_vals
            int idx = 0;
            for(int i = 0; i < theta0_vals.Length; i++)
            {
                for (int k = 0; k < theta1_vals.Length; k++)
                {
                    Matrix<double> t = Matrix<double>.Build.Dense(2, 1);
                    t[0, 0] = theta0_vals[i];
                    t[1, 0] = theta1_vals[k];

                    sx[idx] = theta0_vals[i];
                    sy[idx] = theta1_vals[k];
                    sz[idx] = ComputeCost(X, y, t);
                    idx++;
                }
            }

            GnuPlot.HoldOn();
            GnuPlot.Set("terminal wxt 1");
            GnuPlot.Set("title \"Cost function J\"");
            GnuPlot.Set("key bottom right");
            GnuPlot.Set("xlabel \"{/Symbol q}_0\"");
            GnuPlot.Set("ylabel \"{/Symbol q}_1\"");

            // surface plot
            GnuPlot.SPlot(sx, sy, sz, "title \"J({/Symbol q}_0,{/Symbol q}_1)\"");

            // Contour plot
            GnuPlot.Set("terminal wxt 2");
            GnuPlot.Set("cntrparam levels 12", "logscale z", "isosamples 2", "xr[-10:10]","yr[-1:4]"); 
            GnuPlot.Unset("key", "label");
            GnuPlot.Contour(sx, sy, sz);

            GnuPlot.Plot(new double[]{theta[0,0]}, new double[]{theta[1,0]}, "pt 2 ps 1 lc rgb \"red\"");
        }


        private static void Pause()
        {
            if(!System.Console.IsOutputRedirected)
            {
                Console.WriteLine("Program paused. Press enter to continue.\n");
                Console.ReadKey();
            }
        }
        private static void PlotLinearFit(double[] x, double[] h)
        {
            
            GnuPlot.Plot(x, h, "with lines linestyle 1 linewidth 1 title \"Linear Regression\" ");
        }

        // plot the data
        public static void PlotData(double[] x, double[] y)
        {
            GnuPlot.Set("title \"Linear Regression\"");
            GnuPlot.Set("key bottom right");
            //GnuPlot.Set("key title \"Legend\"");
            GnuPlot.Set("xlabel \"Population of City in 10,000s\"");
            GnuPlot.Set("ylabel \"Profit in $10,000s\"");
            GnuPlot.Plot(x, y, "pt 2 ps 1 lc rgb \"red\" title \"Training data\"");
        }

        // compute cost function J
        public static double ComputeCost(Matrix<double> X, Matrix<double> y, Matrix<double> theta)
        {
            int m = y.RowCount;
            Matrix<double> J;

            J = 1.0/(2.0*m) * (X * theta - y).Transpose() * (X * theta - y);

            return J[0,0];

        }

        // compute gradient descent
        public static (Matrix<double> theta, Matrix<double> J_history) GradientDescent(Matrix<double> X, Matrix<double> y, Matrix<double> theta, double alpha, int num_iters)
        {
                        
            int m = y.RowCount; // number of training examples
            Matrix<double> J_history = Matrix<double>.Build.Dense(num_iters, 1);

            for(int i = 0; i<num_iters; i++)
            {
                J_history[i,0] = ComputeCost(X, y, theta);
                theta = theta - (alpha/m) * X.Transpose() * (X * theta - y);
            }
            return (theta, J_history);
        }
    }
}
