using System;
using System.Collections.Generic;
using System.Globalization;
using MathNet.Numerics.Data.Matlab;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace ex7_pca
{
    class Program
    {
        static void Main(string[] args)
        {
            if(!System.Console.IsOutputRedirected)
                System.Console.Clear();

            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

            System.Console.WriteLine("Principal Component Analysis ex.7_pca");
            System.Console.WriteLine("=====================================\n");

            var M = Matrix<double>.Build;
            var V = Vector<double>.Build;

            //// ================== Part 1: Load Example Dataset  ===================
            //  We start this exercise by using a small dataset that is easily to
            //  visualize
            //

            // read all matrices of a file by name into a dictionary
            Dictionary<string,Matrix<double>> mr = MatlabReader.ReadAll<double>("data\\ex7data1.mat");

            // loads dataset
            System.Console.WriteLine("Loading dataset....\n");
            Matrix<double> X = mr["X"];
            System.Console.WriteLine(X);

            //// =============== Part 2: Principal Component Analysis ===============
            //  You should now implement PCA, a dimension reduction technique. You
            //  should complete the code in pca.m
            //

            System.Console.WriteLine("\nRunning PCA on example dataset.\n\n");

            //  Before running PCA, it is important to first normalize X
            System.Console.WriteLine("Features normalization...");
            (Matrix<double> X_norm, Matrix<double> mu, Matrix<double> sigma) norm_res;
            norm_res = FeatureNormalize(X);

            System.Console.WriteLine(norm_res.X_norm);

            //  Run PCA
            (Matrix<double> U, Vector<double> S) pca_res;
            pca_res = pca(norm_res.X_norm);
            
            System.Console.WriteLine("Top eigenvector: \n");
            System.Console.WriteLine(" U(:,1) = {0:f6} {1:f6} \n", pca_res.U[0,0], pca_res.U[1,0]);
            System.Console.WriteLine("\n(you should expect to see -0.707107 -0.707107)\n");            

            Pause();
        }

        private static (Matrix<double> U, Vector<double> S) pca(Matrix<double> X)
        {
            int m = X.RowCount;
            int n = X.ColumnCount;

            // covariance matrix
            Matrix<double> sigma = (X.Transpose() * X) / m;

            var svd = sigma.Svd();

            (Matrix<double> U, Vector<double> S) res;
            res.U = svd.U;
            res.S = svd.S;
            return res;
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
