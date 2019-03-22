using System;
using System.Linq;
using System.Globalization;
using AwokeKnowing.GnuplotCSharp;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using MathNet.Numerics.Data.Matlab;
using System.Collections.Generic;
using System.Drawing;

namespace ex3
{
    class Program
    {
        static void Main(string[] args)
        {
            if(!System.Console.IsOutputRedirected)
                System.Console.Clear();

            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

            System.Console.WriteLine("Multi-class Classiﬁcation and Neural Networks ex.3");
            System.Console.WriteLine("================================================\n");

            var M = Matrix<double>.Build;
            var V = Vector<double>.Build;

            // read all matrices of a file by name into a dictionary
            Dictionary<string,Matrix<double>> ms = MatlabReader.ReadAll<double>("data\\ex3data1.mat");
            
            Matrix<double> X = ms["X"];
            Vector<double> y = ms["y"].Column(0);

            // get a casual sequence of 100 int numbers
            var srs = new MathNet.Numerics.Random.SystemRandomSource();
            var seq = srs.NextInt32Sequence(0, 5000).Take(100).ToList();

            // Randomly select 100 data points to display
            Vector<double>[] sel = new Vector<double>[100];
            int idx = 0;
            Vector<double> v = V.Dense(400);
            foreach(int i in seq) {
                sel[idx++] = X.Row(i);
            }

            // display
            DisplayData(sel);
            
            Pause();

            //// ============ Part 2a: Vectorize Logistic Regression ============
            //  In this part of the exercise, you will reuse your logistic regression
            //  code from the last exercise. You task here is to make sure that your
            //  regularized logistic regression implementation is vectorized. After
            //  that, you will implement one-vs-all classification for the handwritten
            //  digit dataset.
            //

            // Test case for lrCostFunction
            System.Console.WriteLine("\nTesting Cost Function with regularization");

            Vector<double> theta_t = V.DenseOfArray(new[]{-2.0, -1, 1, 2});

            Matrix<double> X_t = M.DenseOfArray(new [,] {
                {1.0, 0.1, 0.6, 1.1},
                {1.0, 0.2, 0.7, 1.2},
                {1.0, 0.3, 0.8, 1.3},
                {1.0, 0.4, 0.9, 1.4},
                {1.0, 0.5, 1.0, 1.5},                                                
            });
            Vector<Double> y_t = V.DenseOfArray(new []{1.0,0,1,0,1});
            int lambda_t = 3;

            LogisticRegression lr = new LogisticRegression(X_t, y_t);
            lr.Lambda = lambda_t;
            double J = lr.Cost(theta_t);
            Vector<double> grad = lr.Gradient(theta_t);

            System.Console.WriteLine("\nCost: {0:f5}\n", J);
            System.Console.WriteLine("Expected cost: 2.534819\n");
            System.Console.WriteLine("Gradients:\n");
            System.Console.WriteLine(" {0:f5} \n", grad);
            System.Console.WriteLine("Expected gradients:\n");
            System.Console.WriteLine(" 0.146561\n -0.548558\n 0.724722\n 1.398003\n");

            Pause();

            //// ============ Part 2b: One-vs-All Training ============
            System.Console.WriteLine("\nTraining One-vs-All Logistic Regression...\n");

            double lambda = 0.1;
            int num_labels = 10;
            Matrix<double> all_theta = OneVsAll(X, y, num_labels, lambda);

            Pause();

            // ================ Part 3: Predict for One-Vs-All ================

            Vector<double> pred = PredictOneVsAll(all_theta, X);
            Vector<double> comp = V.Dense(y.Count);

            for(int i = 0; i < y.Count; i++)
            {
                if(pred[i] == y[i])
                    comp[i] = 1;
                else
                    comp[i] = 0;
            }


            double accuracy = comp.Mean() * 100;
            System.Console.WriteLine("\nTraining Set Accuracy: {0:f5}\n", accuracy);

            Pause();

        }

        private static Vector<double> PredictOneVsAll(Matrix<double> all_theta, Matrix<double> X)
        {
            int m = X.RowCount;

            // Add ones to the X data matrix
            X = X.InsertColumn(0, Vector<double>.Build.Dense(m, 1.0));

            Vector<double> pred = Vector<double>.Build.Dense(m);

            var h = LogisticRegression.Sigmoid(X * all_theta.Transpose());
            
            for(int i = 0; i < m; i++)
            {
                pred[i] = h.Row(i).MaximumIndex() + 1;
            }

            return pred;
        }

        private static Matrix<double> OneVsAll(Matrix<double> X, Vector<double> y, int num_labels, double lambda)
        {
            int m = X.RowCount;         // num. of examples
            int n = X.ColumnCount;      // num of features

            Matrix<double> all_theta = Matrix<double>.Build.Dense(num_labels, n + 1);  // return matrix of theta

            // Add ones to the X data matrix
            X = X.InsertColumn(0, Vector<double>.Build.Dense(m, 1.0));

            LogisticRegression lr = new LogisticRegression(X, null);
            var obj = ObjectiveFunction.Gradient(lr.Cost, lr.Gradient);
            var solver = new BfgsMinimizer(1e-3, 1e-3, 1e-3, 100);    
            MinimizationResult result;

            for(int c = 1; c <= num_labels; c++)
            {
                Vector<double> initial_theta = Vector<double>.Build.Dense(n + 1);

                // set 1 where y==c, 0 otherwise
                Vector<double> y_t = y.Map(d => (d == (double)c ? 1.0 : 0));

                lr.y = y_t;
                lr.Lambda = lambda;
                result = solver.FindMinimum(obj, initial_theta);
                all_theta.SetRow(c - 1, result.MinimizingPoint);
                System.Console.WriteLine("Label: {0} | Iterations: {1} | Cost: {2:f5} ", c, result.Iterations, result.FunctionInfoAtMinimum.Value);

            }

            System.Console.WriteLine();
            return all_theta;
        }

        // convert bitmpa in NTSC (YIQ)
        private static Matrix<double> BitmapToNYSC(Bitmap bmp)
        {
            int w = bmp.Width;
            int h = bmp.Height;

            Matrix<double> m = Matrix<double>.Build.Dense(w * h, 3);

            int i = 0;
            for(int x = 0; x < bmp.Width; x++)
            {
                for(int y = 0; y < bmp.Height; y++)
                {
                    var p = bmp.GetPixel(x, y);
                    double Y = 0.299 * p.R + 0.587 * p.G + 0.114 * p.B;
                    double I = 0.596 * p.R - 0.275 * p.G - 0.321 * p.B;
                    double Q = 0.212 * p.R - 0.523 * p.G + 0.311 * p.B;
                    m[i,0] = Y;
                    m[i,1] = I;
                    m[i,2] = Q;
                    i++;
                }
            }

            return m;
        }
        private static void DisplayData(Vector<double>[] X)
        {
            int w = 20;
            int h = 20;
            int row = 0;
            int col = 0;
            int offsetRow = 0;
            int offsetCol = 0;

            double[,] d = new double[h * 10, w * 10];


            for(int i = 0; i < X.Length; i++)
            {
                for(int k = 0; k < (w * h); k ++)
                {
                    d[row + offsetRow, col + offsetCol] = X[i][k];
                    offsetCol++;

                    if(offsetCol % w == 0)
                    {
                        offsetRow++;
                        offsetCol = 0;
                    }
                }

                col += w;
                if(col >= (w * 10))
                {
                    row += h;
                    col  = 0;
                }

                offsetRow = 0;
                offsetCol = 0;
            }

            //d = RotateRight(d);
            
            //GnuPlot.Unset("key");
            GnuPlot.Unset("colorbox");
            //GnuPlot.Unset("tics");
            GnuPlot.Set("grid");
            GnuPlot.Set("palette grey");
            //GnuPlot.Set("xrange [0:200]");
            //GnuPlot.Set("yrange [0:200]");
            GnuPlot.Set("pm3d map");
            GnuPlot.SPlot(d, "with image");

        }
        private static double[,] RotateRight(double [,] image)
        {

            int rows = image.GetLength(0);
            int cols = image.GetLength(1);
            double[,] d = new double[rows, cols];

            for(int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    d[rows-c-1, r] = image[r,c];
                }
            }

            return d;
        }

        private static double[,] RotateLeft(double [,] image)
        {

            int rows = image.GetLength(0);
            int cols = image.GetLength(1);
            double[,] d = new double[rows, cols];

            for(int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    d[c, cols-r-1] = image[r,c];
                }
            }

            return d;
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
