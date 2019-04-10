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


namespace ex4
{
    public class NeuralNetwork
    {
        public Matrix<double> X { get; }
        public Vector<double> y { get; }
        public double Lambda { get; set; }

        public int OutputLayerSize { get; }
        public int HiddenLayerSize { get; }
        public int InputLayerSize { get; }

        private Matrix<double> labels;

        public NeuralNetwork(Matrix<double> X, Vector<double> y, int inputLayerSize, int hiddenLayerSize, int outputLayerSize)
        {
            this.X = X;
            this.y = y;    
            this.Lambda = 0;
            this.InputLayerSize = inputLayerSize;
            this.HiddenLayerSize = hiddenLayerSize;
            this.OutputLayerSize = outputLayerSize;

            this.labels = RecodeLabels(this.y);

        }

        private Matrix<double> RecodeLabels(Vector<double> y)
        {
            // ------------------------------------------------------------  
            // recode y(5000x1) --> y(50000x10);
            // ex.:  5 -> [0 0 0 0 1 0 0 0 0 0]
            // ------------------------------------------------------------

            int m = y.Count;
            Matrix<double> recoded = Matrix<double>.Build.Dense(m, this.OutputLayerSize);

            for(int i = 0; i < m; i++)
            {
                switch(this.y[i])
                {
                    case 1:
                        recoded[i,0] = 1;
                        break;
                    case 2:
                        recoded[i,1] = 1;
                        break;
                    case 3:
                        recoded[i,2] = 1;
                        break;
                    case 4:
                        recoded[i,3] = 1;
                        break;
                    case 5:
                        recoded[i,4] = 1;
                        break;
                    case 6:
                        recoded[i,5] = 1;
                        break;
                    case 7:
                        recoded[i,6] = 1;
                        break;
                    case 8:
                        recoded[i,7] = 1;
                        break;
                    case 9:
                        recoded[i,8] = 1;
                        break;
                    case 10:
                        recoded[i,9] = 1;
                        break;

                }
            }

            return recoded;
        }

        public Vector<double> Predict(Vector<double> theta, Matrix<double> X)
        {
            int m = X.RowCount;   // number of training examples

            // unroll the parameters into matrix

            // theta1: 25 x 401
            // theta2: 10 x 26
            (Matrix<double> theta1, Matrix<double> theta2) param = ReshapeParameters(theta);            

            // ---------------------------------
            // feed forward
            // ---------------------------------
            Matrix<double> a1, a2, a3, z2, z3, h;
            a1 = X;                                                                 // 5000 x 400
            a1 = a1.InsertColumn(0, Vector<double>.Build.Dense(a1.RowCount, 1.0));  // add ones: 5000 x 401
            z2 = param.theta1 * a1.Transpose();                                     // (25x401) x (401x5000) = 25 x 5000
            a2 = Sigmoid(z2.Transpose());                                           // 5000 x 25
            a2 = a2.InsertColumn(0, Vector<double>.Build.Dense(a2.RowCount, 1.0));  // add ones: 5000 x 26
            z3 = param.theta2 * a2.Transpose();                                     // (10x26) x (26x5000) = 10 x 5000
            a3 = Sigmoid(z3);                                                       // 10 x 5000;
            h = a3.Transpose();                                                     // 5000 x 10 (hypothesys)
            
            Vector<double> pred = Vector<double>.Build.Dense(m);

            for(int i = 0; i < m; i++)
            {
                pred[i] = h.Row(i).MaximumIndex() + 1;
            }

            return pred;

        }

        public double Cost(Vector<double> theta)
        {
            int m = X.RowCount;   // number of training examples

            // unroll the parameters into matrix

            // theta1: 25 x 401
            // theta2: 10 x 26
            (Matrix<double> theta1, Matrix<double> theta2) param = ReshapeParameters(theta);            

            double J = 0.0; // the return value

            // ---------------------------------
            // feed forward
            // ---------------------------------
            Matrix<double> a1, a2, a3, z2, z3, h;
            a1 = X;                                                                 // 5000 x 400
            a1 = a1.InsertColumn(0, Vector<double>.Build.Dense(a1.RowCount, 1.0));  // add ones: 5000 x 401
            z2 = param.theta1 * a1.Transpose();                                     // (25x401) x (401x5000) = 25 x 5000
            a2 = Sigmoid(z2.Transpose());                                           // 5000 x 25
            a2 = a2.InsertColumn(0, Vector<double>.Build.Dense(a2.RowCount, 1.0));  // add ones: 5000 x 26
            z3 = param.theta2 * a2.Transpose();                                     // (10x26) x (26x5000) = 10 x 5000
            a3 = Sigmoid(z3);                                                       // 10 x 5000;
            h = a3.Transpose();                                                     // 5000 x 10 (hypothesys)
            
            // ------------------------------------------------------------
            // cost function (vectorized)
            // ------------------------------------------------------------
            for(int i = 0; i < this.OutputLayerSize; i++)
            {
                Vector<double> hi = h.Column(i);
                Vector<double> yi = this.labels.Column(i);

                J = J + (1.0/m) * (-yi * Vector<double>.Log(hi) - (1-yi) * Vector<double>.Log(1 - hi)); 
            }

            // ------------------------------------------------------------
            // regularization
            // ------------------------------------------------------------

            // remove bias
            Matrix<double> theta1, theta2;
            theta1 = param.theta1.RemoveColumn(0);      // 25 x 400
            theta2 = param.theta2.RemoveColumn(0);      // 10 x 25

            double reg = 0.0;
            reg += theta1.PointwisePower(2).RowSums().Sum();
            reg += theta2.PointwisePower(2).RowSums().Sum();
            reg = reg * Lambda / (2.0 * m);

            J += reg;

            return J;

        }

        public Vector<double> Gradient(Vector<double> theta)
        {
            int m = X.RowCount;   // number of training examples

            // unroll the parameters into matrix

            // theta1: 25 x 401
            // theta2: 10 x 26
            (Matrix<double> theta1, Matrix<double> theta2) param = ReshapeParameters(theta);            

            // ---------------------------------
            // feed forward
            // ---------------------------------
            Matrix<double> a1, a2, a3, z2, z3, h;
            a1 = X;                                                                 // 5000 x 400
            a1 = a1.InsertColumn(0, Vector<double>.Build.Dense(a1.RowCount, 1.0));  // add ones: 5000 x 401
            z2 = param.theta1 * a1.Transpose();                                     // (25x401) * (401x5000) = 25 x 5000
            a2 = Sigmoid(z2.Transpose());                                           // 5000 x 25
            a2 = a2.InsertColumn(0, Vector<double>.Build.Dense(a2.RowCount, 1.0));  // add ones: 5000 x 26
            z3 = param.theta2 * a2.Transpose();                                     // (10x26) * (26x5000) = 10 x 5000
            a3 = Sigmoid(z3);                                                       // 10 x 5000;
            h = a3.Transpose();                                                     // 5000 x 10 (hypothesys)

            Vector<double> delta3;      // 10 x 1
            Vector<double> delta2;      // 26 x 1 (bias included)
            Matrix<double> DELTA2;      // 10 x 26
            Matrix<double> DELTA1;      // 25 x 401

            DELTA2 = Matrix<double>.Build.Dense(param.theta2.RowCount, param.theta2.ColumnCount);
            DELTA1 = Matrix<double>.Build.Dense(param.theta1.RowCount, param.theta1.ColumnCount);

            // process one example at the time 
            for(int i = 0; i < m; i++)
            {
                // errors in nodes of layer 3 
                delta3 = h.Row(i) - this.labels.Row(i);         // 10 x 1

                // errors in nodes of layer 2
                delta2 = (param.theta2.Transpose() * delta3);                           // (26x10) * (10x1) = 26 x 1
                delta2 = Vector<double>.Build.DenseOfArray(delta2.Skip(1).ToArray());   // remove bias: 25x1)
                delta2 = delta2.PointwiseMultiply(SigmoidGradient(z2.Column(i)));       // (25x1) .* (25x1) = 25 x 1

                DELTA2 = DELTA2 + delta3.ToColumnMatrix() * a2.Row(i).ToRowMatrix();    // (10x1) * (1x26)  = (10x26)
                DELTA1 = DELTA1 + delta2.ToColumnMatrix() * a1.Row(i).ToRowMatrix();    // (25x1) * (1x401) = (25x401)
                
            }

            Matrix<double> theta1_grad = (1.0/m) * DELTA1;                              // 25 x 401
            Matrix<double> theta2_grad = (1.0/m) * DELTA2;                              // 10 x 26

            // --------------------------------------------------------------------
            // regularization
            // --------------------------------------------------------------------

            param.theta1.MapIndexedInplace((r, c, d) => c == 0 ? 0.0 : d);    // exclude 1st column (theta0) which contains the bias term
            param.theta2.MapIndexedInplace((r, c, d) => c == 0 ? 0.0 : d);    // exclude 1st column (theta0) which contains the bias term

            theta1_grad = theta1_grad + (this.Lambda / m) * param.theta1;
            theta2_grad = theta2_grad + (this.Lambda / m) * param.theta2;

            // unroll gradient
            Vector<double> grad = UnrollParameters(theta1_grad, theta2_grad);
            
            return grad;
        }

        public Vector<double> SigmoidGradient(Vector<double> theta)
        {
            
            Vector<double> gz = Sigmoid(theta);

            // sigmoid derivative
            Vector<double> grad = gz.PointwiseMultiply(1 -gz);
            return grad;
        }

        public Matrix<double> SigmoidGradient(Matrix<double> theta)
        {
            
            Matrix<double> gz = Sigmoid(theta);

            // sigmoid derivative
            Matrix<double> grad = gz.PointwiseMultiply(1 -gz);
            return grad;
        }

        public static Matrix<double> Sigmoid(Matrix<double> z)
        {
                return 1/(1+Matrix<double>.Exp(-z));
        }

        public static Vector<double> Sigmoid(Vector<double> z)
        {
                return 1/(1+Vector<double>.Exp(-z));
        }

        public static Vector<double> UnrollParameters(Matrix<double> theta1, Matrix<double> theta2)
        {
            Vector<double> p = Vector<double>.Build.DenseOfArray(
                theta1.ToColumnMajorArray()
                .Concat(theta2.ToColumnMajorArray())
                .ToArray()
                );

            return p;
        }

        public static Matrix<double> ReshapeParameters(Vector<double> vec, int m, int n)
        {
            Matrix<double> r = Matrix<double>.Build.DenseOfColumnMajor(m, n, vec);
            return r;
        }

        public (Matrix<double> theta1, Matrix<double> theta2) ReshapeParameters(Vector<double> theta)
        {
            // Reshape theta back into the parameters Theta1 and Theta2, the weight matrices
            // for our 2 layer neural network

            // theta1: 25 X 401
            Matrix<double> theta1 = Matrix<double>.Build.DenseOfColumnMajor(
                this.HiddenLayerSize, 
                this.InputLayerSize + 1, 
                theta.SubVector(0, this.HiddenLayerSize * (InputLayerSize + 1))                
                );



            // theta2: 10 X 26
            Matrix<double> theta2 = Matrix<double>.Build.DenseOfColumnMajor(
                this.OutputLayerSize, 
                this.HiddenLayerSize + 1, 
                theta.SubVector(this.HiddenLayerSize * (this.InputLayerSize + 1), this.OutputLayerSize * (this.HiddenLayerSize + 1))                
                );

            return (theta1, theta2);
        }

    }
}