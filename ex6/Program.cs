using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using AwokeKnowing.GnuplotCSharp;
using libsvm;
using MathNet.Numerics.Data.Matlab;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace ex6
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


            //// =============== Part 1: Loading and Visualizing Data ================
            //  We start the exercise by first loading and visualizing the dataset. 
            //  The following code will load the dataset into your environment and plot
            //  the data.
            //

            System.Console.WriteLine("Loading and Visualizing Data ...\n");

            // Load from ex6data1: 
            // You will have X, y in your environment
            Dictionary<string,Matrix<double>> ms = MatlabReader.ReadAll<double>("data\\ex6data1.mat");

            Matrix<double> X = ms["X"];                 // 51 X 2
            Vector<double> y = ms["y"].Column(0);       // 51 X 1

            // Plot training data
            GnuPlot.HoldOn();
            PlotData(X, y);

            Pause();

            //// ==================== Part 2: Training Linear SVM ====================
            //  The following code will train a linear SVM on the dataset and plot the
            //  decision boundary learned.
            //

            System.Console.WriteLine("\nTraining Linear SVM ...\n");

            // You should try to change the C value below and see how the decision
            // boundary varies (e.g., try C = 1000)
            double C = 1.0;
            var linearKernel = KernelHelper.LinearKernel();

            List<List<double>> libSvmData = ConvertToLibSvmFormat(X, y);
            svm_problem prob = ProblemHelper.ReadProblem(libSvmData);                        
            var svc = new C_SVC(prob, linearKernel, C);
            
            PlotBoundary(X, svc);
            GnuPlot.HoldOff();

            System.Console.WriteLine();            
            
            Pause();

            //// =============== Part 3: Implementing Gaussian Kernel ===============
            //  You will now implement the Gaussian kernel to use
            //  with the SVM. You should complete the code in gaussianKernel.m
            //

            System.Console.WriteLine("\nEvaluating the Gaussian Kernel ...\n");

            double sigma = 2.0;
            double sim = GaussianKernel(
                V.DenseOfArray(new [] {1.0, 2, 1}),
                V.DenseOfArray(new [] {0.0, 4, -1}),
                sigma
            );

            System.Console.WriteLine("Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {0:f6} :\n\t{1:f6}\n(for sigma = 2, this value should be about 0.324652)\n", sigma, sim);

            Pause();

            //// =============== Part 4: Visualizing Dataset 2 ================
            //  The following code will load the next dataset into your environment and 
            //  plot the data. 
            //

            System.Console.WriteLine("Loading and Visualizing Data ...\n");

            // Load from ex6data2: 
            // You will have X, y in your environment
            ms = MatlabReader.ReadAll<double>("data\\ex6data2.mat");

            X = ms["X"];                 // 863 X 2
            y = ms["y"].Column(0);       // 863 X 1

            // Plot training data
            GnuPlot.HoldOn();
            PlotData(X, y);

            Pause();

            //// ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
            //  After you have implemented the kernel, we can now use it to train the 
            //  SVM classifier.
            // 

            System.Console.WriteLine("\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n");

            // SVM Parameters
            C = 1; 
            sigma = 0.1;
            double gamma = 1/(2*sigma*sigma);  

            var rbfKernel = KernelHelper.RadialBasisFunctionKernel(gamma);

            libSvmData = ConvertToLibSvmFormat(X, y);
            prob = ProblemHelper.ReadProblem(libSvmData);                                    
            svc = new C_SVC(prob, rbfKernel, C);
            

            PlotBoundary(X, svc);
            GnuPlot.HoldOff();

            Pause();

            double acc = svc.GetCrossValidationAccuracy(10);
            System.Console.WriteLine("\nCross Validation Accuracy: {0:f6}\n", acc);

            Pause();

            //// =============== Part 6: Visualizing Dataset 3 ================
            //  The following code will load the next dataset into your environment and 
            //  plot the data. 
            //

            System.Console.WriteLine("Loading and Visualizing Data ...\n");

            // Load from ex6data2: 
            // You will have X, y in your environment
            ms = MatlabReader.ReadAll<double>("data\\ex6data3.mat");

            Matrix<double> Xval;
            Vector<double> yval;

            X = ms["X"];                 // 211 X 2
            y = ms["y"].Column(0);       // 211 X 1
            Xval = ms["Xval"];           // 200 X 2
            yval = ms["yval"].Column(0); // 200 X 1

            // Plot training data
            GnuPlot.HoldOn();
            PlotData(X, y);

            //// ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

            //  This is a different dataset that you can use to experiment with. Try
            //  different values of C and sigma here.
            // 


            (C, sigma) = Dataset3Params(X, y, Xval, yval);

            gamma = 1/(2*sigma*sigma);  
            rbfKernel = KernelHelper.RadialBasisFunctionKernel(gamma);

            libSvmData = ConvertToLibSvmFormat(X, y);
            prob = ProblemHelper.ReadProblem(libSvmData);                                    
            svc = new C_SVC(prob, rbfKernel, C);            

            PlotBoundary(X, svc);

            GnuPlot.HoldOff();
            Pause();
        }

        private static (double C, double sigma) Dataset3Params(Matrix<double> x, Vector<double> y, Matrix<double> xval, Vector<double> yval)
        {
            double[] c_val = new []{0.01,0.03,0.1,0.3,1,3,10,3};         // possibili valori di C
            double[] sigma_test = new []{0.01,0.03,0.1,0.3,1,3,10,3};   // possibili valori di sigma
            
            // Results:
            //  [:,0] - error
            //  [:,1] - C
            //  [:,2] -  sigma
            Matrix<double> results = Matrix<double>.Build.Dense(c_val.Length * sigma_test.Length, 3);

            // convert x, y in libsvm format
            List<List<double>> libSvmData = ConvertToLibSvmFormat(x, y);

            // try all possible pairs of C and sigma
            int i = 0;
            foreach(double c_temp in c_val)
            {   
                foreach(double s_temp in sigma_test)
                {
                    double gamma = 1/(2 * s_temp * s_temp);  
                    var rbfKernel = KernelHelper.RadialBasisFunctionKernel(gamma);

                    svm_problem prob = ProblemHelper.ReadProblem(libSvmData);                                    
                    C_SVC svc = new C_SVC(prob, rbfKernel, c_temp);
                    
                    double error = ComputeValidationError(svc, xval, yval);

                    results[i, 0] = error;
                    results[i, 1] = c_temp;
                    results[i, 2] = s_temp;
                    i++;
                }
            }

            int idx = results.Column(0).MinimumIndex();

            return (results.Row(idx)[1],results.Row(idx)[2]);
        }

        private static double ComputeValidationError(C_SVC svc, Matrix<double> xval, Vector<double> yval)
        {
            int m = xval.RowCount;
            double errorCount = 0;

            for(int i = 0; i < m; i++)
            {
                    svm_node n1 = new svm_node();
                    n1.index = 1;
                    n1.value = xval.Row(i)[0];

                    svm_node n2 = new svm_node();
                    n2.index = 2;
                    n2.value = xval.Row(i)[1];
                    
                    double pred = svc.Predict(new [] {n1, n2}); 

                    if(pred != yval[i])
                        errorCount++;

            }


            return errorCount / m;
        }

        private static double GaussianKernel(Vector<double> x1, Vector<double> x2, double sigma)
        {
            double sim = 0;

            sim = ((x1 - x2).PointwisePower(2)).Sum();
            sim = Math.Exp(-sim/(2*sigma*sigma));
            return sim;
        }

        private static void PlotBoundary(Matrix<double> x, C_SVC svc)
        {
            double min = x.Column(0).Min();
            double max = x.Column(0).Max();

            double[] x0 = MathNet.Numerics.Generate.LinearSpaced(100, min, max);

            min = x.Column(1).Min();
            max = x.Column(1).Max();

            double[] x1 = MathNet.Numerics.Generate.LinearSpaced(100, min, max);

            int size = x0.Length * x1.Length;
            double[] sx = new double[size];
            double[] sy = new double[size];
            double[] sz = new double[size];

            int idx = 0;
            for(int i = 0; i < x0.Length; i++)
            {
                for(int j = 0; j < x1.Length; j++)
                {
                    sx[idx] = x0[i];
                    sy[idx] = x1[j];
                   
                    svm_node n1 = new svm_node();
                    n1.index = 1;
                    n1.value = x0[i];

                    svm_node n2 = new svm_node();
                    n2.index = 2;
                    n2.value = x1[j];
                    
                    double z = svc.Predict(new [] {n1, n2}); 
                    sz[idx] = z;
                    idx++;

                }
            }

            GnuPlot.Set("cntrparam levels discrete 0.5"); 
            GnuPlot.Contour(sx, sy, sz, "title \"Decision Boundary\"");
        }

        private static List<List<double>> ConvertToLibSvmFormat(Matrix<double> x, Vector<double> y)
        {
            List<List<double>> data = new List<List<double>>();

            int m = x.RowCount;

            for(int i = 0; i < m; i++)
            {
                List<double> r = new List<double>();

                r.Add(y[i]);
                foreach(var c in x.Row(i))
                {
                    r.Add(c);
                }
                data.Add(r);
            }

            return data;
        }

        public static void PlotData(Matrix<double> X, Vector<double> y)
        {

            List<(double x1,double x2)> pos = new List<(double x1, double x2)>();
            List<(double x1,double x2)> neg = new List<(double x1, double x2)>();

            for(int i = 0; i<y.Count; i++)
            {
                if(y[i] == 1)
                    pos.Add((X[i,0], X[i,1]));
                else
                    neg.Add((X[i,0], X[i,1]));
            }

                        
            GnuPlot.HoldOn();
            GnuPlot.Plot(pos.Select(p => p.x1).ToArray(), pos.Select(p => p.x2).ToArray(), "pt 1 ps 1 lc rgb \"black\" notitle");            
            GnuPlot.Plot(neg.Select(p => p.x1).ToArray(), neg.Select(p => p.x2).ToArray(), "pt 7 ps 1 lc rgb \"yellow\" notitle");
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
