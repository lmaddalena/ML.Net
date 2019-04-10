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
using MathNet.Numerics.Distributions;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;

namespace ex4
{
    class Program
    {
        static void Main(string[] args)
        {
            if(!System.Console.IsOutputRedirected)
                System.Console.Clear();

            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

            System.Console.WriteLine("Multi-class Classification and Neural Networks ex.4");
            System.Console.WriteLine("===================================================\n");

            var M = Matrix<double>.Build;
            var V = Vector<double>.Build;

            // Setup the parameters you will use for this exercise
            int input_layer_size  = 400;  // 20x20 Input Images of Digits
            int hidden_layer_size = 25;   // 25 hidden units
            int num_labels = 10;          // 10 labels, from 1 to 10   
                                          // (note that we have mapped "0" to label 10)

            //  =========== Part 1: Loading and Visualizing Data =============
            //  We start the exercise by first loading and visualizing the dataset. 
            //  You will be working with a dataset that contains handwritten digits.
            //

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

            // ================ Part 2: Loading Parameters ================
            // In this part of the exercise, we load some pre-initialized 
            // neural network parameters.

            System.Console.WriteLine("\nLoading Saved Neural Network Parameters ...\n");

            // read all matrices of a file by name into a dictionary
            Dictionary<string,Matrix<double>> mr = MatlabReader.ReadAll<double>("data\\ex3weights.mat");
            
            Matrix<double> theta1 = mr["Theta1"];      // 25 X 401
            Matrix<double> theta2 = mr["Theta2"];      // 10 X 26

            // Unroll parameters 
            Vector<double> nn_params = NeuralNetwork.UnrollParameters(theta1, theta2);

            Pause();

            //  ================ Part 3: Compute Cost (Feedforward) ================
            //  To the neural network, you should first start by implementing the
            //  feedforward part of the neural network that returns the cost only. You
            //  should complete the code in nnCostFunction.m to return cost. After
            //  implementing the feedforward to compute the cost, you can verify that
            //  your implementation is correct by verifying that you get the same cost
            //  as us for the fixed debugging parameters.
            //
            //  We suggest implementing the feedforward cost *without* regularization
            //  first so that it will be easier for you to debug. Later, in part 4, you
            //  will get to implement the regularized cost.


            System.Console.WriteLine("\nFeedforward Using Neural Network ...\n");

            // Weight regularization parameter (we set this to 0 here).

            NeuralNetwork nn = new NeuralNetwork(X, y, input_layer_size, hidden_layer_size, num_labels);
            nn.Lambda = 0.0;
            double J = nn.Cost(nn_params);

            System.Console.WriteLine("Cost at parameters (loaded from ex4weights): {0:f6}\n(this value should be about 0.287629)\n", J);
            Pause();

            // =============== Part 4: Implement Regularization ===============
            // Once your cost function implementation is correct, you should now
            // continue to implement the regularization with the cost.
            //

            System.Console.WriteLine("\nChecking Cost Function (w/ Regularization) ... \n");

            // Weight regularization parameter (we set this to 1 here).
            nn.Lambda = 1.0;

            J = nn.Cost(nn_params);

            System.Console.WriteLine("Cost at parameters (loaded from ex4weights): {0:f6} \n(this value should be about 0.383770)\n", J);
            Pause();

            // ================ Part 5: Sigmoid Gradient  ================
            //  Before you start implementing the neural network, you will first
            //  implement the gradient for the sigmoid function. You should complete the
            //  code in the sigmoidGradient.m file.
            
            System.Console.WriteLine("\nEvaluating sigmoid gradient...\n");

            var g = nn.SigmoidGradient(V.DenseOfArray(new[]{-1.0, -0.5, 0, 0.5, 1}));
            System.Console.WriteLine("Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ");
            System.Console.WriteLine("{0:f5} ", g);
            System.Console.WriteLine("\n\n");

            Pause();


            // ================ Part 6: Initializing Pameters ================
            // In this part of the exercise, you will be starting to implment a two
            // layer neural network that classifies digits. You will start by
            // implementing a function to initialize the weights of the neural network
            // (randInitializeWeights.m)

            System.Console.WriteLine("\nInitializing Neural Network Parameters ...\n");

            Matrix<double> initial_Theta1 = RandInitializeWeights(input_layer_size + 1, hidden_layer_size);
            Matrix<double> initial_Theta2 = RandInitializeWeights(hidden_layer_size + 1, num_labels);

            // Unroll parameters 
            Vector<double> initial_nn_params = NeuralNetwork.UnrollParameters(initial_Theta1, initial_Theta2);

            Pause();
            // =============== Part 7: Implement Backpropagation ===============
            // Once your cost matches up with ours, you should proceed to implement the
            // backpropagation algorithm for the neural network. You should add to the
            // code you've written in nnCostFunction.m to return the partial
            // derivatives of the parameters.
            
            System.Console.WriteLine("\nChecking Backpropagation... \n");
            CheckGradient(0);

            Pause();

            // =============== Part 8: Implement Regularization ===============
            //  Once your backpropagation implementation is correct, you should now
            //  continue to implement the regularization with the cost and gradient.
            
            System.Console.WriteLine("\nChecking Backpropagation (w/ Regularization) ... \n");

            //  Check gradients by running checkNNGradients
            double lambda = 3;
            CheckGradient(lambda);

            // Also output the costFunction debugging values
            nn.Lambda = lambda;
            double debug_J  = nn.Cost(nn_params);

            System.Console.WriteLine("\n\nCost at (fixed) debugging parameters (w/ lambda = {0:f1}): {1:f6} " +
                    "\n(for lambda = 3, this value should be about 0.576051)\n\n", lambda, debug_J);

            Pause();

            // =================== Part 8: Training NN ===================
            //  You have now implemented all the code necessary to train a neural 
            //  network. To train your neural network, we will now use "fmincg", which
            //  is a function which works similarly to "fminunc". Recall that these
            //  advanced optimizers are able to train our cost functions efficiently as
            //  long as we provide them with the gradient computations.

            System.Console.WriteLine("\nTraining Neural Network... \n");

            //  After you have completed the assignment, change the MaxIter to a larger
            //  value to see how more training helps.
            int maxIter = 40;

            //  You should also try different values of lambda
            lambda = 1;
            nn.Lambda = lambda;

            
            var obj = ObjectiveFunction.Gradient(nn.Cost, nn.Gradient);
            var solver = new LimitedMemoryBfgsMinimizer(1e-5, 1e-5, 1e-5, 5, maxIter);                        
            var result = solver.FindMinimum(obj, initial_nn_params);
            System.Console.WriteLine("Reason For Exit: {0}", result.ReasonForExit);
            System.Console.WriteLine("Iterations: {0}", result.Iterations);
            System.Console.WriteLine("Cost: {0:e}", result.FunctionInfoAtMinimum.Value);            

            
            Pause();

            // ================= Part 10: Implement Predict =================
            //  After training the neural network, we would like to use it to predict
            //  the labels. You will now implement the "predict" function to use the
            //  neural network to predict the labels of the training set. This lets
            //  you compute the training set accuracy.

            Vector<double> pred = nn.Predict(result.MinimizingPoint, X);
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

        private static void CheckGradient(double lamda)
        {
            int input_layer_size = 3;
            int hidden_layer_size = 5;
            int num_labels = 3;
            int m = 5;

            // We generate some 'random' test data
            Matrix<double> theta1 = DebugInitializeWeights(hidden_layer_size, input_layer_size);
            Matrix<double> theta2 = DebugInitializeWeights(num_labels, hidden_layer_size);

            Matrix<double> X  = DebugInitializeWeights(m, input_layer_size - 1);
            //Vector<double> y = Vector<double>.Build.DenseOfArray(new []{2.0, 3, 1, 2, 3});            
            Vector<double> y = Vector<double>.Build.Dense(m);

            for(int i = 0; i < m; i++)
            {
                y[i] = (i % num_labels) + 1;
            }

            // Unroll parameters
            Vector<double> nn_params = NeuralNetwork.UnrollParameters(theta1, theta2);

            NeuralNetwork nn = new NeuralNetwork(X, y, input_layer_size, hidden_layer_size, num_labels);
            nn.Lambda = lamda;

            double cost = nn.Cost(nn_params);
            Vector<double> grad = nn.Gradient(nn_params);
            Vector<double> numGrad = NumericalGradient(nn.Cost, nn_params);

            for(int i = 0; i < grad.Count; i++)
            {
                System.Console.WriteLine("{0:f10}\t{1:f10}", grad[i], numGrad[i]);
            }

            System.Console.WriteLine("\nThe above two columns you get should be very similar.\n" +
                                     "(Right-Your Numerical Gradient, Left-Analytical Gradient)\n\n");

            double diff =  (numGrad - grad).Norm(2) / (numGrad + grad).Norm(2);

            System.Console.WriteLine("If your backpropagation implementation is correct, then \n" +
                                        "the relative difference will be small (less than 1e-9). \n" +
                                        "\nRelative Difference: {0:e}\n", diff);
        }

        // Initialize the weights of a layer with fan_in
        // incoming connections and fan_out outgoing connections using a fixed
        //strategy, this will help you later in debugging        
        private static Matrix<double> DebugInitializeWeights(int fanOut, int fanIn)
        {
            Matrix<double> W = Matrix<double>.Build.Dense(fanOut, 1 + fanIn);

            int i = 1;

            for(int c = 0; c < W.ColumnCount; c++)
            {
                for(int r = 0; r < W.RowCount; r++)
                {
                    W[r, c] = Math.Sin(i) / 10.0;
                    i++;                    
                }
            }

            return W;
        }

        private static Vector<double> NumericalGradient(Func<Vector<double>, double> costFunc, Vector<double> theta)
        {
            double J = costFunc(theta);
            double epsilon = 1e-4;

            Vector<double> thetaPlus;
            Vector<double> thetaMinus;
            Vector<double> gradApprox = Vector<double>.Build.Dense(theta.Count);

            for(int i = 0; i < theta.Count; i++)
            {
                thetaPlus = theta.Clone();
                thetaMinus = theta.Clone();                
                thetaPlus[i] = thetaPlus[i] + epsilon;
                thetaMinus[i] = thetaMinus[i] - epsilon;
                gradApprox[i] = (costFunc(thetaPlus) - costFunc(thetaMinus))/ (2*epsilon);
            }

            return gradApprox;
        }

        // test cost function and gradient
        private static void TestCostAndGradient()
        {
            System.Console.WriteLine("Test cost function and gradient\n");

            Matrix<double> X = Matrix<double>.Build.DenseOfArray(new [,] {
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0}
            });
            X = Matrix<double>.Cos(X);

            Vector<double> y = Vector<double>.Build.DenseOfArray(new [] {4.0, 2, 3});

            Vector<double> param = Vector<double>.Build.DenseOfArray(Generate.LinearRange(1, 1, 18));
            param = param / 10.0;

            NeuralNetwork nn = new NeuralNetwork(X, y, 2, 2, 4);
            nn.Lambda = 4;

            double J = nn.Cost(param);
            System.Console.WriteLine("Cost: {0:f3} \n(this value should be about 19.474)\n", J);

            Vector<double> grad = nn.Gradient(param);
            System.Console.WriteLine("Gradient:\n");
            System.Console.WriteLine(grad);
        }

        private static Matrix<double> RandInitializeWeights(int rows, int columns)
        {
            var srs = new MathNet.Numerics.Random.SystemRandomSource();
            IContinuousDistribution dist = new MathNet.Numerics.Distributions.Normal(srs);            
            double epsilon = 0.12;
            //Matrix<double> w = Matrix<double>.Build.Random(rows, columns) * (2 * epsilon) - epsilon;
            
            var seq = srs.NextDoubleSequence().Take(rows*columns);
            Matrix<double> w = Matrix<double>.Build.DenseOfColumnMajor(rows, columns, seq);
            w = w  * (2 * epsilon) - epsilon;
            return w;
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
