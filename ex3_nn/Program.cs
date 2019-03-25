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


namespace ex3_nn
{
    class Program
    {
        static void Main(string[] args)
        {
            if(!System.Console.IsOutputRedirected)
                System.Console.Clear();

            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

            System.Console.WriteLine("Neural Networks ex.3_nn");
            System.Console.WriteLine("==================================================\n");

            var M = Matrix<double>.Build;
            var V = Vector<double>.Build;

            //// =========== Part 1: Loading and Visualizing Data =============
            //   We start the exercise by first loading and visualizing the dataset. 
            //   You will be working with a dataset that contains handwritten digits.
            //

            // Load Training Data
            System.Console.WriteLine("Loading and Visualizing Data ...\n");
            
            // read all matrices of a file by name into a dictionary
            Dictionary<string,Matrix<double>> mr = MatlabReader.ReadAll<double>("data\\ex3data1.mat");
            
            Matrix<double> X = mr["X"];
            Vector<double> y = mr["y"].Column(0);

            Double m = X.RowCount;

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

            // ================ Part 2: Loading Pameters ================
            // In this part of the exercise, we load some pre-initialized 
            // neural network parameters.

            System.Console.WriteLine("\nLoading Saved Neural Network Parameters ...\n");

            // read all matrices of a file by name into a dictionary
            mr = MatlabReader.ReadAll<double>("data\\ex3weights.mat");
            
            Matrix<double> theta1 = mr["Theta1"];      // 25 X 401
            Matrix<double> theta2 = mr["Theta2"];      // 10 X 26

            Pause();

            //// ================= Part 3: Implement Predict =================
            //  After training the neural network, we would like to use it to predict
            //  the labels. You will now implement the "predict" function to use the
            //  neural network to predict the labels of the training set. This lets
            //  you compute the training set accuracy.

            Vector<double> pred = Predict(theta1, theta2, X);

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

            //  Randomly permute examples
            seq = srs.NextInt32Sequence(0, 5000).Take(5000).ToList();

            for(int i = 0; i < m; i++)
            {
                // display
                DisplayData(new[] {X.Row(seq[i])} );

                Matrix<double> x = M.DenseOfRowVectors(new[] {X.Row(seq[i])});
                pred = Predict(theta1, theta2, x);
                double p = pred[0];
                System.Console.WriteLine("\nNeural Network Prediction: {0:N0} (digit {1:N0})\n", p, p%10);

                // Pause with quit option
                System.Console.WriteLine("Paused - press enter to continue, q to exit:");
                string s = Console.ReadLine();
                if (s.ToLower() == "q")
                    break;

            }

            Pause();

        }

        private static Vector<double> Predict(Matrix<double> theta1, Matrix<double> theta2, Matrix<double> X)
        {

            int m = X.RowCount;

            // add ones
            X = X.InsertColumn(0, Vector<double>.Build.Dense(m, 1));

            // size theta1:   25 x 401
            // size theta2:   10 x 26
            // size X:      5000 x 401

            var z1 = X * theta1.Transpose();            // z1: 5000 X 25
            var a1 = LogisticRegression.Sigmoid(z1);    // a1: 5000 X 25

            // add ones
            a1 = a1.InsertColumn(0, Vector<double>.Build.Dense(m, 1));     // a1: 5000 X 26
            var z2 = a1 * theta2.Transpose();           // z2: 5000 X 10
            var a2 = LogisticRegression.Sigmoid(z2);    // a2: 5000 X 10

            Vector<double> pred = Vector<double>.Build.Dense(m);

            for(int i = 0; i < m; i++)
            {
                pred[i] = a2.Row(i).MaximumIndex() + 1;
            }

            return pred;
        }

        private static void DisplayData(Vector<double>[] X)
        {
            int w = 20;
            int h = 20;
            int row = 0;
            int col = 0;
            int offsetRow = 0;
            int offsetCol = 0;
            int dim = (int)Math.Sqrt(X.Length);

            double[,] d = new double[h * dim, w * dim];


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
                if(col >= (w * dim))
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
