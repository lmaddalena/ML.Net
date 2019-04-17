using System;
using System.Collections.Generic;
using System.Globalization;
using AwokeKnowing.GnuplotCSharp;
using MathNet.Numerics;
using MathNet.Numerics.Data.Matlab;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using MathNet.Numerics.Statistics;

namespace ex5
{
    class Program
    {
        static void Main(string[] args)
        {
            if(!System.Console.IsOutputRedirected)
                System.Console.Clear();

            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

            System.Console.WriteLine("Regularized Linear Regression and Bias v.s. Variance ex.5");
            System.Console.WriteLine("=========================================================\n");

            var M = Matrix<double>.Build;
            var V = Vector<double>.Build;

            // =========== Part 1: Loading and Visualizing Data =============
            // We start the exercise by first loading and visualizing the dataset. 
            // The following code will load the dataset into your environment and plot
            // the data.


            // Load Training Data
            System.Console.WriteLine("Loading and Visualizing Data ...\n");

            // Load from ex5data1: 
            // You will have X, y, Xval, yval, Xtest, ytest in your environment
            Dictionary<string,Matrix<double>> ms = MatlabReader.ReadAll<double>("data\\ex5data1.mat");
            
            Matrix<double> X = ms["X"];
            Vector<double> y = ms["y"].Column(0);
            Matrix<double> Xval = ms["Xval"];
            Vector<double> yval = ms["yval"].Column(0);
            Matrix<double> Xtest = ms["Xtest"];
            Vector<double> ytest = ms["ytest"].Column(0);

            // m = Number of examples
            int m = X.RowCount;

            GnuPlot.HoldOn();
            PlotData(X.Column(0).ToArray(), y.ToArray());

            Pause();

            // =========== Part 2: Regularized Linear Regression Cost =============
            //  You should now implement the cost function for regularized linear 
            //  regression. 
            Vector<double> theta = V.Dense(2, 1.0);
            LinearRegression lr = new LinearRegression(X.InsertColumn(0, V.Dense(m, 1)), y, 1);
            double J = lr.Cost(theta);
            Vector<double> grad = lr.Gradient(theta);

            System.Console.WriteLine("Cost at theta = [1 ; 1]: {0:f6} \n(this value should be about 303.993192)\n", J);

            Pause();

            // =========== Part 3: Regularized Linear Regression Gradient =============
            //  You should now implement the gradient for regularized linear 
            //  regression.

            System.Console.WriteLine("Gradient at theta = [1 ; 1]:  [{0:f6}; {1:f6}] \n(this value should be about [-15.303016; 598.250744])\n", grad[0], grad[1]);

            Pause();

            // =========== Part 4: Train Linear Regression =============
            //  Once you have implemented the cost and gradient correctly, the
            //  trainLinearReg function will use your cost function to train 
            //  regularized linear regression.
            // 
            //  Write Up Note: The data is non-linear, so this will not give a great 
            //                 fit.
            //

            //  Train linear regression with lambda = 0
            lr = new LinearRegression(X.InsertColumn(0, V.Dense(m, 1)), y, 0);
            var result = lr.Train();

            Vector<double> h = X.InsertColumn(0, V.Dense(m, 1)) * result.MinimizingPoint;  // hypothesys
            PlotLinearFit(X.Column(0).ToArray(), h.ToArray());
            GnuPlot.HoldOff();

            Pause();

            // =========== Part 5: Learning Curve for Linear Regression =============
            //  Next, you should implement the learningCurve function. 
            //
            //  Write Up Note: Since the model is underfitting the data, we expect to
            //                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
            //
            (Vector<double> error_train, Vector<double> error_val) res;
            res = LearningCurve(X.InsertColumn(0, V.Dense(m, 1)), y, Xval.InsertColumn(0, V.Dense(Xval.RowCount, 1)), yval, 0);
            PlotLinearLearningCurve(
                Generate.LinearRange(1, 1, m),
                res.error_train.ToArray(),
                res.error_val.ToArray()
            );
            
            System.Console.WriteLine("# Training Examples\tTrain Error\tCross Validation Error\n");

            for(int i = 0; i < m; i++)
            {
                System.Console.WriteLine("\t{0,2}\t\t{1:f6}\t{2:f6}", i, res.error_train[i], res.error_val[i]);
            }
            System.Console.WriteLine();

            Pause();

            // =========== Part 6: Feature Mapping for Polynomial Regression =============
            //  One solution to this is to use polynomial regression. You should now
            //  complete polyFeatures to map each example into its powers
            //

            int p = 8;

            // Map X onto Polynomial Features and Normalize
            Matrix<double> X_poly = MapPolyFeatures(X, p);

            // normalize
            var norm = FeatureNormalize(X_poly);
            X_poly = norm.X_norm;

            // add one's
            X_poly = X_poly.InsertColumn(0, V.Dense(X_poly.RowCount, 1));
            
            // Map X_poly_test and normalize (using mu and sigma)
            Matrix<double> X_poly_test = MapPolyFeatures(Xtest, p);
            for(int i = 0; i < X_poly_test.ColumnCount; i++)
            {
                Vector<double> v = X_poly_test.Column(i);
                v = v - norm.mu[0, i];
                v = v / norm.sigma[0, i];
                X_poly_test.SetColumn(i, v);
            }

            // add one's
            X_poly_test = X_poly_test.InsertColumn(0, V.Dense(X_poly_test.RowCount, 1));

            // Map X_poly_val and normalize (using mu and sigma)
            Matrix<double> X_poly_val = MapPolyFeatures(Xval, p);
            for(int i = 0; i < X_poly_val.ColumnCount; i++)
            {
                Vector<double> v = X_poly_val.Column(i);
                v = v - norm.mu[0, i];
                v = v / norm.sigma[0, i];
                X_poly_val.SetColumn(i, v);
            }

            // add one's
            X_poly_val = X_poly_val.InsertColumn(0, V.Dense(X_poly_val.RowCount, 1));

            System.Console.WriteLine("Normalized Training Example 1:\n");
            System.Console.WriteLine(X_poly.Row(0));

            Pause();

            // =========== Part 7: Learning Curve for Polynomial Regression =============
            //  Now, you will get to experiment with polynomial regression with multiple
            //  values of lambda. The code below runs polynomial regression with 
            //  lambda = 0. You should try running the code with different values of
            //  lambda to see how the fit and learning curve change.
            //

            double lambda = 0;
            lr = new LinearRegression(X_poly, y, lambda);
            var minRes = lr.Train();

            GnuPlot.HoldOn();
            GnuPlot.Set("terminal wxt 1");
            PlotData(X.Column(0).ToArray(), y.ToArray());
            PlotFit(X.Column(0).Minimum(), X.Column(0).Maximum(), norm.mu.Row(0), norm.sigma.Row(0), minRes.MinimizingPoint, p, lambda);            
            GnuPlot.HoldOff();

            // learning curve
            GnuPlot.Set("terminal wxt 2");

            res = LearningCurve(X_poly, y, X_poly_val, yval, lambda);
            PlotLinearLearningCurve(
                Generate.LinearRange(1, 1, m),
                res.error_train.ToArray(),
                res.error_val.ToArray()
            );
            
            System.Console.WriteLine("# Training Examples\tTrain Error\tCross Validation Error\n");

            for(int i = 0; i < m; i++)
            {
                System.Console.WriteLine("\t{0,2}\t\t{1:f6}\t{2:f6}", i, res.error_train[i], res.error_val[i]);
            }
            System.Console.WriteLine();

            Pause();

            // =========== Part 8: Validation for Selecting Lambda =============
            //  You will now implement validationCurve to test various values of 
            //  lambda on a validation set. You will then use this to select the
            //  "best" lambda value.
            //
            var resVal = ValidationCurve(X_poly, y,  X_poly_val, yval);
            PlotValidationCurve(resVal.Lamda_vec.ToArray(), resVal.error_train.ToArray(), resVal.error_val.ToArray());

            System.Console.WriteLine("# Lambda\tTrain Error\tCross Validation Error\n");

            for(int i = 0; i < resVal.error_train.Count; i++)
            {
                System.Console.WriteLine("\t{0:f4}\t\t{1:f6}\t{2:f6}", resVal.Lamda_vec[i], resVal.error_train[i], resVal.error_val[i]);
            }
            System.Console.WriteLine();


            Pause();

            // =========== Part 9: Compute test set error =============
            // Compute the test error using the best value of λ you 
            // found        
            lambda = 3.0;
            lr = new LinearRegression(X_poly, y, lambda);
            minRes = lr.Train();
            theta = minRes.MinimizingPoint;

            h = X_poly_test * theta;
            m = X_poly.RowCount;

            System.Console.WriteLine("Evaluating test-set:\n");
            System.Console.WriteLine("# \tHypothesis\tExpected\tError\n");
            for(int i = 0; i < m; i++)
            {
                System.Console.WriteLine("{0,3}\t{1:f6}\t{2:f6}\t{3:f6}", i+1, h[i], ytest[i], h[i]-ytest[i]);                
            }
            
            double mae = (h - ytest).L1Norm();      // Mean Absolute Error            
            double mse = (h - ytest).L2Norm();      // Mean Squared Error
            double rmse = Math.Sqrt(mse);           // Root Mean Squared Error

            System.Console.WriteLine("\nMAE on test set: {0:F6}", mae);
            System.Console.WriteLine("MSE on test set: {0:F6}", mse);
            System.Console.WriteLine("RMSE on test set: {0:F6}\n", rmse);                    

            Pause();

            GnuPlot.HoldOn();
            GnuPlot.Set("terminal wxt 3");
            PlotData(Xtest.Column(0).ToArray(), ytest.ToArray());
            PlotFit(Xtest.Column(0).Minimum(), Xtest.Column(0).Maximum(), norm.mu.Row(0), norm.sigma.Row(0), theta, p, lambda);            
            GnuPlot.HoldOff();

            Pause();
        }

        private static void PlotValidationCurve(double[] x, double[] jtrain, double[] jvc)
        {
            GnuPlot.HoldOn();
            GnuPlot.Set("title \"Validation curve\"");
            GnuPlot.Set("xlabel \"Lamda\"");
            GnuPlot.Set("key top right box");
            GnuPlot.Set("ylabel \"Error\"");
            GnuPlot.Set("autoscale xy");

            GnuPlot.Plot(x, jtrain, "with lines ls 1 lw 2 lc rgb \"cyan\" title \"Train\" ");
            GnuPlot.Plot(x, jvc, "with lines ls 1 lw 2 lc rgb \"orange\" title \"Cross Validation\" ");

            GnuPlot.HoldOff();
        }


        private static void PlotFit(double min_x, double max_x, Vector<double> mu, Vector<double> sigma, Vector<double> theta, int p, double lambda)
        {
            // Plot training data and fit
            var x = Matrix<double>.Build.DenseOfColumnArrays(Generate.LinearRange(min_x - 15, .05, max_x + 25));

            Matrix<double> X_poly = MapPolyFeatures(x, p);
            for(int i = 0; i < X_poly.ColumnCount; i++)
            {
                Vector<double> v = X_poly.Column(i);
                v = v - mu[i];
                v = v / sigma[i];
                X_poly.SetColumn(i, v);
            }
            
            X_poly = X_poly.InsertColumn(0, Vector<double>.Build.Dense(X_poly.RowCount, 1));
            var hypothesis = X_poly * theta;

            GnuPlot.Set(string.Format("title \"Polynomial Regression Fit (lambda = {0:f6})\"", lambda));
            GnuPlot.Set("xlabel \"Change in water level (x)\"");
            GnuPlot.Set("ylabel \"Water flowing out of the dam (y)\"");
            GnuPlot.Set("autoscale xy");
            //GnuPlot.Set("xr[-100:100]");
            //GnuPlot.Set("yr[-200:400]");

            GnuPlot.Plot(x.Column(0).ToArray(), hypothesis.ToArray(), "with lines dashtype 2 lw 1 lc rgb \"blue\" notitle");

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
                // column vector
                Vector<double> v = x.Column(c);

                // compute mean and stdev per column
                double mu = v.Mean();
                double sigma = v.StandardDeviation();


                // column vector normalization
                v = (v - mu) / sigma;

                // assigna computed values to result
                res.mu[0, c] = mu;
                res.sigma[0, c] = sigma;                                
                res.X_norm.SetColumn(c, v);
            }

            return res;
        }
        private static (Vector<double> error_train, Vector<double> error_val) LearningCurve(
            Matrix<double> X, Vector<double> y, Matrix<double> xVal, Vector<double> yVal, double lambda){

                int m = X.RowCount;
                Vector<double> theta = Vector<double>.Build.Dense(X.ColumnCount, 0);

                Vector<double> error_train = Vector<double>.Build.Dense(m, 0);
                Vector<double> error_val = Vector<double>.Build.Dense(m, 0);

                LinearRegression lr = new LinearRegression();

                MinimizationResult result;


                for(int i = 0; i < m; i++)
                {
                    var xset = X.SubMatrix(0, i+1, 0, X.ColumnCount);
                    var yset = y.SubVector(0, i+1);

                    lr.X = xset;
                    lr.y = yset;
                    lr.Lambda = lambda;
                    
                    result = lr.Train();
                    System.Console.WriteLine("Iteration {0,5} | Cost: {1:e}", result.Iterations, result.FunctionInfoAtMinimum.Value);

                    lr.Lambda = 0;
                    error_train[i] = lr.Cost(result.MinimizingPoint);

                    
                    lr.X = xVal;
                    lr.y = yVal;
                    lr.Lambda = 0;
                    error_val[i] = lr.Cost(result.MinimizingPoint);
                }

                System.Console.WriteLine();
                return (error_train, error_val);

        }

        private static (Vector<double> Lamda_vec, Vector<double> error_train, Vector<double> error_val) ValidationCurve(
            Matrix<double> X, Vector<double> y, Matrix<double> xVal, Vector<double> yVal)
        {
            Vector<double> lamda_vec = Vector<double>.Build.DenseOfArray(new [] { 
                0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10
            });

            int m = lamda_vec.Count;

            Vector<double> theta = Vector<double>.Build.Dense(X.ColumnCount, 0);
            Vector<double> error_train = Vector<double>.Build.Dense(m, 0);
            Vector<double> error_val = Vector<double>.Build.Dense(m, 0);

            LinearRegression lr = new LinearRegression(X, y, 0);
            MinimizationResult result;

            for(int i = 0; i < m; i++)
            {
                lr.Lambda = lamda_vec[i];
                lr.X = X;
                lr.y = y;

                result = lr.Train();
                System.Console.WriteLine("Iteration {0,5} | Cost: {1:e}", result.Iterations, result.FunctionInfoAtMinimum.Value);


                lr.Lambda = 0;
                error_train[i] = lr.Cost(result.MinimizingPoint);

                lr.X = xVal;             
                lr.y = yVal;   
                lr.Lambda = 0;
                error_val[i] = lr.Cost(result.MinimizingPoint);

            }

            return (lamda_vec, error_train, error_val);
        }        

        private static void PlotLinearLearningCurve(double[] x, double[] jtrain, double[] jvc)
        {
            GnuPlot.HoldOn();
            GnuPlot.Set("title \"Learning curve for linear regression\"");
            GnuPlot.Set("xlabel \"Number of training examples\"");
            GnuPlot.Set("key top right box");
            GnuPlot.Set("ylabel \"Error\"");
            GnuPlot.Set("xr[0:13]");
            GnuPlot.Set("yr[0:150]");

            GnuPlot.Plot(x, jtrain, "with lines ls 1 lw 2 lc rgb \"cyan\" title \"Train\" ");
            GnuPlot.Plot(x, jvc, "with lines ls 1 lw 2 lc rgb \"orange\" title \"Cross Validation\" ");

            GnuPlot.HoldOff();
        }

        // Map original feature X in polynomial features of grade p
        private static Matrix<double> MapPolyFeatures(Matrix<double> X, int p)
        {
            int m = X.RowCount;
            int n = p;
            Matrix<double> feat = Matrix<double>.Build.Dense(m, n);

            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j < n; j++)
                {   
                    feat[i, j] = Math.Pow(X[i, 0], j + 1);
                }
            }
            
            return feat;

        }

        private static void PlotLinearFit(double[] x, double[] h)
        {
            
            GnuPlot.Plot(x, h, "with lines linestyle 1 linewidth 1 title \"Linear Regression\" ");
        }


        public static void PlotData(double[] x, double[] y)
        {
            GnuPlot.Set("title \"Water flowing\"");
            GnuPlot.Set("key bottom right");
            GnuPlot.Set("xlabel \"Change in water level (x)\"");
            GnuPlot.Set("ylabel \"Water flowing out of the dam (y)\"");
            GnuPlot.Plot(x, y, "pt 2 ps 1 lc rgb \"red\" notitle");
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
