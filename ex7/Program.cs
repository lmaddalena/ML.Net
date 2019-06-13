using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using AwokeKnowing.GnuplotCSharp;
using MathNet.Numerics.Data.Matlab;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace ex7
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

            //// ================= Part 1: Find Closest Centroids ====================
            //  To help you implement K-Means, we have divided the learning algorithm 
            //  into two functions -- findClosestCentroids and computeCentroids. In this
            //  part, you should complete the code in the findClosestCentroids function. 
            //
            
            System.Console.WriteLine("Finding closest centroids.\n\n");

            // Load an example dataset that we will be using
            Dictionary<string,Matrix<double>> ms = MatlabReader.ReadAll<double>("data\\ex7data2.mat");
            Matrix<double> X = ms["X"];                 // 300 X 2

            System.Console.WriteLine(X);

            // Select an initial set of centroids
            int K = 3;                                  // 3 Centroids
            Matrix<double> initial_centroids = M.DenseOfArray(new [,] {
                { 3.0, 3.0 },
                { 6.0, 2.0 },
                { 8.0, 5.0 }
            });

            System.Console.Write("Initial centroids: ");
            System.Console.WriteLine(initial_centroids);

            Vector<double> idx = FindClosestCentroids(X, initial_centroids);


            System.Console.WriteLine("Closest centroids for the first 3 examples: \n");
            System.Console.WriteLine(idx.SubVector(0, 3));
            System.Console.WriteLine("\n(the closest centroids should be 0, 2, 1 respectively)\n");

            Pause();

            //// ===================== Part 2: Compute Means =========================
            //  After implementing the closest centroids function, you should now
            //  complete the computeCentroids function.
            //
            System.Console.WriteLine("\nComputing centroids means.\n\n");

            //  Compute means based on the closest centroids found in the previous part.
            Matrix<double> centroids;
            centroids = ComputeCentroids(X, idx, K);

            System.Console.WriteLine("Centroids computed after initial finding of closest centroids: \n");
            System.Console.WriteLine(centroids);
            System.Console.WriteLine("\nthe centroids should be");
            System.Console.WriteLine("   [ 2.428301 3.157924 ]");
            System.Console.WriteLine("   [ 5.813503 2.633656 ]");
            System.Console.WriteLine("   [ 7.119387 3.616684 ]\n");

            Pause();

            //// =================== Part 3: K-Means Clustering ======================
            //  After you have completed the two functions computeCentroids and
            //  findClosestCentroids, you have all the necessary pieces to run the
            //  kMeans algorithm. In this part, you will run the K-Means algorithm on
            //  the example dataset we have provided. 
            //
            System.Console.WriteLine("\nRunning K-Means clustering on example dataset.\n\n");

            // Settings for running K-Means
            K = 3;
            int max_iters = 10;

            // For consistency, here we set centroids to specific values
            // but in practice you want to generate them automatically, such as by
            // settings them to be random examples (as can be seen in
            // kMeansInitCentroids).
            initial_centroids = M.DenseOfArray(new [,] {
                { 3.0, 3.0 },
                { 6.0, 2.0 },
                { 8.0, 5.0 }
            });

            // Run K-Means algorithm. The 'true' at the end tells our function to plot
            // the progress of K-Means
            (Matrix<double> centroids, Vector<double> idx) kMeans = RunkMeans(X, initial_centroids, max_iters, true);
            System.Console.WriteLine("\nK-Means Done.\n\n");
            Pause();
        }

        private static (Matrix<double> centroids, Vector<double> idx) RunkMeans(Matrix<double> X, Matrix<double> initial_centroids, int max_iters, bool plot_progress)
        {
            // Plot the data if we are plotting progress
            if(plot_progress)
            {
                GnuPlot.HoldOn();
            }

            // Initialize values
            int m = X.RowCount;
            int n = X.ColumnCount;
            int K = initial_centroids.RowCount;
            Matrix<double> centroids = initial_centroids;
            Matrix<double> previous_centroids = centroids;
            Vector<double> idx = Vector<double>.Build.Dense(m);

            // Run K-Means
            for(int i = 0; i < max_iters; i++)
            {
                // Output progress
                System.Console.WriteLine("K-Means iteration {0}/{1}...", i+1, max_iters);

                // For each example in X, assign it to the closest centroid
                idx = FindClosestCentroids(X, centroids);

                // Optionally, plot progress here
                if(plot_progress)
                {
                    PlotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
                    previous_centroids = centroids;
                    //Pause();
                }

                // Given the memberships, compute new centroids
                centroids = ComputeCentroids(X, idx, K);
            }

            if(plot_progress)
            {
                GnuPlot.HoldOff();
            }

            return (centroids, idx);
        }

        private static void PlotProgresskMeans(Matrix<double> X, Matrix<double> centroids, Matrix<double> previous_centroids, Vector<double> idx, int K, int i)
        {
            GnuPlot.Set($"title \"Iteration number {i+1}");
            PlotDataPoints(X, idx, K);

            // Plot the centroids as black x's
            double[] x1 = centroids.Column(0).ToArray();
            double[] x2 = centroids.Column(1).ToArray();
            GnuPlot.Plot(x1, x2, "pt 2 ps 1 lw 3 lc rgb \"black\" notitle");

            // Plot the history of the centroids with lines
            for(int j = 0; j < centroids.RowCount; j++)
            {
                double[] x = new double[] { centroids[j, 0], previous_centroids[j, 0]};
                double[] y = new double[] { centroids[j, 1], previous_centroids[j, 1]};
                GnuPlot.Plot(x, y, "with lines linestyle 1 linewidth 2 notitle");               
            }

        }

        private static void PlotDataPoints(Matrix<double> X, Vector<double> idx, int K)
        {

            double[] x1 = X.Column(0).ToArray();
            double[] x2 = X.Column(1).ToArray();
            string[] colors = {"red","green","cyan","blue","purple","yellow","orange"};
            string color = colors[0];
            Matrix<double> sel = null;

            for(int k = 0; k < K; k++)
            {
                color = colors[k];
                sel = SelectFromIndexes(X, FindIndexes(idx, k));
                x1 = sel.Column(0).ToArray();
                x2 = sel.Column(1).ToArray();

                GnuPlot.Plot(x1, x2, $"pt 6 ps .7 lc rgb \"{color}\" notitle");

            }            

        }

        private static Matrix<double> ComputeCentroids(Matrix<double> X, Vector<double> idx, int K)
        {
            // Useful variables
            int m = X.RowCount;
            int n = X.ColumnCount;

            Matrix<double> centroids = Matrix<double>.Build.Dense(K, n);

            // ====================== YOUR CODE HERE ======================
            // Instructions: Go over every centroid and compute mean of all points that
            //               belong to it. Concretely, the row vector centroids(i, :)
            //               should contain the mean of the data points assigned to
            //               centroid i.
            //
            // Note: You can use a for-loop over the centroids to compute this.
            //
           
            for(int k = 0; k < K; k++)
            {
                int[] c = FindIndexes(idx, k);
                Matrix<double> X2 = SelectFromIndexes(X, c);
                
                centroids.SetRow(k, X2.ColumnSums() / c.Length);
            }

            return centroids;
        }

        private static Matrix<double> SelectFromIndexes(Matrix<double> x, int[] indexes)
        {
            int m = indexes.Length;
            int n = x.ColumnCount;
            int k = 0;

            Matrix<double> res = Matrix<double>.Build.Dense(m, n);

            foreach(int i in indexes)
            {
                res.SetRow(k++, x.Row(i));
            }

            return res;
        }

        private static int[] FindIndexes(Vector<double> v, int valueToFind)
        {
            int count = v.Count(d => d == valueToFind);
            int[] idx = new int[count];

            int k = 0;

            for(int i = 0; i < v.Count; i++)
            {
                if(v[i] == valueToFind)
                {
                    idx[k++] = i;
                }
            }

            return idx;
        }

        private static Vector<double> FindClosestCentroids(Matrix<double> x, Matrix<double> centroids)
        {           
            // You need to return the following variables correctly. 
            Vector<double> index = Vector<double>.Build.Dense(x.RowCount);   // 300 x 1

            // ====================== YOUR CODE HERE ======================
            // Instructions: Go over every example, find its closest centroid, and store
            //               the index inside idx at the appropriate location.
            //               Concretely, idx(i) should contain the index of the centroid
            //               closest to example i. Hence, it should be a value in the 
            //               range 1..K
            //
            // Note: You can use a for-loop over the examples to compute this.
            //

            int K = centroids.RowCount;                         // number of centroids   
            int m = x.RowCount;                                 // number of training examples
            int N = x.ColumnCount;                              // number of features
            Vector<double> D = Vector<double>.Build.Dense(K);   // vettore con le distanze di ciascun sample i dal centroid k

            for(int i = 0; i < m; i++)
            {
                for(int k = 0; k < K; k++)
                {
                    double d = 0;

                    for(int n = 0; n < N; n++)
                    {
                        d += Math.Pow(x[i,n] -  centroids[k,n], 2); 
                    }

                    D[k] = d;               // distanza del sample i dal centroid k
                }

                int j = D.MinimumIndex();   // j: index of the centroid closest to x(i)
                index[i] = j;
            }

            return index;
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
