using System;
using System.Linq;
using System.Globalization;
using AwokeKnowing.GnuplotCSharp;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using System.Collections.Generic;

namespace ex3
{
    public class LogisticRegression
    {
        public Matrix<double> X { get; set; }
        public Vector<double> y { get; set; }
        public double Lambda { get; set; }
        public LogisticRegression(Matrix<double> X, Vector<double> y)
        {
            this.X = X;
            this.y = y;    
            this.Lambda = 0;
        }

        public double Cost(Vector<double> theta)
        {
            double J = 0;

            double m = this.X.RowCount;         // number of examples
            int n = X.ColumnCount;              // number of features
            
            Vector<double> h = Sigmoid(X * theta);
            J = 1/m * (-y * Vector<double>.Log(h) - (1 - y) * Vector<double>.Log(1 - h));

            // ----------------------------------------
            // regularization
            // ----------------------------------------
            double r = 0;

            // remove theta[0]
            Vector<double> theta_reg = Vector<double>.Build.Dense(theta.Count);
            theta.CopyTo(theta_reg);
            theta_reg[0] = 0;

            r = this.Lambda/(2*m) * (theta_reg * theta_reg);
            J = J + r;
            // ----------------------------------------

            return J;
        }

        public Vector<double> Gradient(Vector<double> theta)
        {
            Vector<double> grad = null;            
            double m = X.RowCount;            
            grad = 1/m * X.Transpose() * (Sigmoid(X*theta) - y);

            // ----------------------------------------
            // regularization
            // ----------------------------------------

            // remove theta[0]
            Vector<double> theta_reg = Vector<double>.Build.Dense(theta.Count);
            theta.CopyTo(theta_reg);
            theta_reg[0] = 0;

            grad += this.Lambda/m * theta_reg;
            // ----------------------------------------

            return grad;
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
        public static Matrix<double> Sigmoid(Matrix<double> z)
        {
                return 1/(1+Matrix<double>.Exp(-z));
        }

    }
}