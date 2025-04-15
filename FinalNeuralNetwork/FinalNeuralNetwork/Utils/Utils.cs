using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Utils
{
    public class Utils
    {
        public static Random Random = new Random();

        public static (double[][], double[][]) GetRandomData(int count)
        {
            var rand = new Random();
            var inputs = new List<double[]>();
            var outputs = new List<double[]>();

            // 1. Generăm inputuri brute
            for (int i = 0; i < count; i++)
            {
                double input1 = rand.NextDouble() * 20 - 10; // Range: [-10, 10]
                double input2 = rand.NextDouble() * 20 - 10;
                inputs.Add(new double[] { input1, input2 });

                double output = 3 * input1 - 2 * input2;
                outputs.Add(new double[] { output });
            }

            // 2. Normalizăm inputurile
            var normalizedInputs = NormalizeInputs(inputs);

            Console.WriteLine($"Generated and normalized {count} samples.");
            return (normalizedInputs, outputs.ToArray());
        }

        private static double[][] NormalizeInputs(List<double[]> data)
        {
            int numFeatures = data[0].Length;
            int numSamples = data.Count;

            double[] means = new double[numFeatures];
            double[] stds = new double[numFeatures];

            // Calculăm media și deviația standard pe fiecare coloană
            for (int i = 0; i < numFeatures; i++)
            {
                var featureColumn = data.Select(row => row[i]).ToArray();
                means[i] = featureColumn.Average();
                stds[i] = Math.Sqrt(featureColumn.Select(x => Math.Pow(x - means[i], 2)).Average());
            }

            // Aplicăm normalizarea (standard score)
            var normalized = new double[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                normalized[i] = new double[numFeatures];
                for (int j = 0; j < numFeatures; j++)
                {
                    normalized[i][j] = (data[i][j] - means[j]) / stds[j];
                }
            }

            return normalized;
        }
    }
}
