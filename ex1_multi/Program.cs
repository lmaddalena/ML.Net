using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.Data.Text;
using System.Globalization;
using AwokeKnowing.GnuplotCSharp;

namespace ex1_multi
{
    class Program
    {
        static void Main(string[] args)
        {
            if(!System.Console.IsOutputRedirected)
                System.Console.Clear();

            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

            var M = Matrix<double>.Build;
            var V = Vector<double>.Build;

            System.Console.WriteLine("Linear Regression ex.1 multiple variables");
            System.Console.WriteLine("=========================================\n");


            // load data
            System.Console.WriteLine("Loading data ...\n");
            Matrix<double> data = DelimitedReader.Read<double>("data\\ex1data2.txt", false, ",", false);            

            Matrix<double> X = data.SubMatrix(0,data.RowCount,0,2);            
            Matrix<double> y = data.SubMatrix(0,data.RowCount,2,1);            
            int m = X.RowCount;

            // Print out some data points
            System.Console.WriteLine("First 10 examples from the dataset: \n");
            var temp = M.DenseOfMatrixArray(new [,]
                {
                    {
                        X.SubMatrix(0,10,0,2),  y.SubMatrix(0,10,0,1)}
                    }
                );

            Console.WriteLine(temp);

            // Scale features and set them to zero mean
            System.Console.WriteLine("Normalizing Features ...\n");

            (Matrix<double> X_norm, Matrix<double> mu, Matrix<double> sigma) norm_res;
            norm_res = FeatureNormalize(X);

            System.Console.WriteLine($"mu: {norm_res.mu}"); 
            System.Console.WriteLine($"sigma: {norm_res.sigma}"); 
            System.Console.WriteLine($"X_norm: {norm_res.X_norm}");

            // Add intercept term to X
            X = norm_res.X_norm;
            X = X.InsertColumn(0, V.Dense(X.RowCount, 1));

            // Running gradient descent ...
            System.Console.WriteLine("Running gradient descent ...\n");

            // Choose some alpha value
            double alpha = 0.01;
            int num_iters = 50;

            GnuPlot.HoldOn();

            Matrix<double> theta = M.Dense(3, 1);
            (Matrix<double> theta, Matrix<double> J_history) res_grad1 = GradientDescentMulti(X, y, theta, alpha, num_iters);
            PlotJ(res_grad1.J_history, "{/Symbol a}=" + alpha, "blue");

            theta = M.Dense(3, 1);
            alpha = 0.03;
            (Matrix<double> theta, Matrix<double> J_history) res_grad2 = GradientDescentMulti(X, y, theta, alpha, num_iters);
            PlotJ(res_grad2.J_history, "{/Symbol a}=" + alpha, "red");

            theta = M.Dense(3, 1);
            alpha = 0.1;
            (Matrix<double> theta, Matrix<double> J_history) res_grad3 = GradientDescentMulti(X, y, theta, alpha, num_iters);
            PlotJ(res_grad3.J_history, "{/Symbol a}=" + alpha, "black");

            theta = M.Dense(3, 1);
            alpha = 0.3;
            (Matrix<double> theta, Matrix<double> J_history) res_grad4 = GradientDescentMulti(X, y, theta, alpha, num_iters);
            PlotJ(res_grad4.J_history, "{/Symbol a}=" + alpha, "green");

            GnuPlot.HoldOff();
            

            // Display gradient descent's result
            System.Console.WriteLine("Theta computed from gradient descent: \n");
            System.Console.WriteLine(res_grad4.theta);
            System.Console.WriteLine("\n");

            // Estimate the price of a 1650 sq-ft, 3 br house
            // Recall that the first column of X is all-ones. Thus, it does
            // not need to be normalized.
            double x1 = 1650;
            double x2 = 3;

            x1 = (x1 - norm_res.mu[0,0]) / norm_res.sigma[0,0];
            x2 = (x2 - norm_res.mu[0,1]) / norm_res.sigma[0,1];

            Matrix<double> K = Matrix<double>.Build.DenseOfRowArrays(new double[]{1, x1, x2});
            System.Console.WriteLine("K is:");
            System.Console.WriteLine(K);
            double price = (K * res_grad4.theta)[0,0];

            System.Console.WriteLine("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${0:f}\n", price);

            Pause();

            // =================================================================
            // NORMAL EQUATION  
            // =================================================================
            System.Console.WriteLine("Solving with normal equations...\n");

            // load data
            System.Console.WriteLine("Loading data ...\n");
            data = DelimitedReader.Read<double>("data\\ex1data2.txt", false, ",", false);            

            X = data.SubMatrix(0,data.RowCount,0,2);            
            y = data.SubMatrix(0,data.RowCount,2,1);            
            m = X.RowCount;

            // Add intercept term to X
            X = X.InsertColumn(0, V.Dense(X.RowCount, 1));

            // Calculate the parameters from the normal equation
            theta = normalEqn(X, y);

            System.Console.WriteLine("Theta computed from the normal equations: \n");
            System.Console.WriteLine(theta);
            System.Console.WriteLine("\n");

            K = Matrix<double>.Build.DenseOfRowArrays(new double[]{1, 1650, 3});
            price = (K * theta)[0,0];

            System.Console.WriteLine("Predicted price of a 1650 sq-ft, 3 br house (using normal equation):\n ${0:f}\n", price);

            Pause();
        }

        // Compute theta using normal equation
        private static Matrix<double> normalEqn(Matrix<double> x, Matrix<double> y)
        {
            Matrix<double> theta;

            theta = (x.Transpose() * x).PseudoInverse() * x.Transpose() * y;
            return theta;
        }

        private static void PlotJ(Matrix<double> j_history, string title, string color)
        {
            double[] x = MathNet.Numerics.Generate.LinearRange(1,1,j_history.RowCount);
            double[] y = j_history.Column(0).ToArray();

            GnuPlot.Set("xlabel \"Number of iteration\"");
            GnuPlot.Set("ylabel \"Cost J\"");
            GnuPlot.Plot(x, y, "with lines linestyle 1 lc rgb \""  + color + "\" linewidth 2 title \"" + title + "\" ");
 
        }

        // Gradient Descent
        private static (Matrix<double> theta, Matrix<double> J_history) GradientDescentMulti(Matrix<double> x, Matrix<double> y, Matrix<double> theta, double alpha, int num_iters)
        {
            double m = y.RowCount;
            Matrix<double> J_history = Matrix<double>.Build.Dense(num_iters, 1);

            for(int i = 0; i < num_iters; i++)
            {

                J_history[i, 0] = ComputeCostMulti(x, y, theta);
                theta = theta - alpha/m * x.Transpose() * (x*theta - y);
            }

            return  (theta, J_history);
        }

        // Compute cost J
        private static double ComputeCostMulti(Matrix<double> x, Matrix<double> y, Matrix<double> theta)
        {
            double j;
            double m = y.RowCount;
            j = 1 / (2.0 * m) * ((x*theta - y).Transpose() * (x*theta - y))[0,0];
            return j;
        }

        private static (Matrix<double> X_norm, Matrix<double> mu, Matrix<double> sigma) FeatureNormalize(Matrix<double> x)
        {
            // Matrix builder
            var M = Matrix<double>.Build;

            // return object
            (Matrix<double> X_norm, Matrix<double> mu, Matrix<double> sigma) res;

            res.mu = M.Dense(1, x.ColumnCount);
            res.sigma = M.Dense(1, x.ColumnCount);
            res.X_norm = M.Dense(x.RowCount, x.ColumnCount);
            
            for(int c = 0; c < x.ColumnCount; c++)
            {
                // compute mean and stdev per column
                double mu = x.Column(c).Mean();
                double sigma = x.Column(c).StandardDeviation();

                // column vector
                Vector<double> v = x.Column(c);

                // column vector normallization
                v = (v - mu) / sigma;

                // assigna computed values to result
                res.mu[0, c] = mu;
                res.sigma[0, c] = sigma;                                
                res.X_norm.SetColumn(c, v);
            }

            return res;   
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
