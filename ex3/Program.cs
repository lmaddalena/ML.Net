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
            Matrix<double> y = ms["y"];

            // get a casual sequence of 100 int numbers
            var srs = new MathNet.Numerics.Random.SystemRandomSource();
            var seq = srs.NextInt32Sequence(0, 5000).Take(100).ToList();

            seq[0] = 4999;
            seq[1] = 1685;
            seq[2] = 3667;
            seq[3] = 4054;
            seq[4] = 1214;

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
