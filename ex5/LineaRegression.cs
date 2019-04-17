using System;
using System.Linq;
using System.Globalization;
using AwokeKnowing.GnuplotCSharp;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using System.Collections.Generic;

namespace ex5
{
    public class LinearRegression
    {
        public Matrix<double> X { get; set; }
        public Vector<double> y { get; set; }
        public double Lambda { get; set; }

        public LinearRegression()
        {}
        
        public LinearRegression(Matrix<double> X, Vector<double> y, double lambda)
        {
            this.X = X;
            this.y = y;    
            this.Lambda = lambda;
        }

        public double Cost(Vector<double> theta)
        {
            double  m = X.RowCount;

            // hypothesis
            Vector<double> h = X * theta;

            // cost
            double J = (h - y) * (h - y) / (2.0 * m);

            // regularization
            var thetaReg = theta.SubVector(1, theta.Count -1);

            J = J + thetaReg * thetaReg * (this.Lambda/(2.0 *m));

            return J;
        }

        public Vector<double> Gradient(Vector<double> theta)
        {
            double  m = X.RowCount;

            // hypothesis
            Vector<double> h = X * theta;

           
            // gradient
            Vector<double> grad = (1.0/m) * (X.Transpose() * (h - y));

            // regularization
            var thetaReg = theta.Clone();
            thetaReg[0] = 0;

            grad = grad + (this.Lambda / m) * thetaReg;

            return grad;

        }

        public MinimizationResult Train()
        {
            Vector<double> theta = Vector<double>.Build.Dense(X.ColumnCount);
            LinearRegression lr = new LinearRegression(this.X, this.y, this.Lambda);
            var obj = ObjectiveFunction.Gradient(lr.Cost, lr.Gradient);
            var solver = new BfgsMinimizer(1e-5, 1e-5, 1e-5, 200);            
            MinimizationResult result = solver.FindMinimum(obj, theta);

            return result;
        }

        public static double Predict(Vector<double> x, Vector<double> theta)
        {
            Vector<double> z = theta.ToRowMatrix() * x;
            double y = Sigmoid(z)[0];
            return y;
        }

        public static Vector<double> Predict(Matrix<double> x, Vector<double> theta)
        {
            Vector<double> z = x * theta;
            Vector<double> y = Sigmoid(z);
            return y;
        }

        public static Vector<double> Sigmoid(Vector<double> z)
        {
                return 1/(1+Vector<double>.Exp(-z));
        }

    }
}