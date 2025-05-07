using FinalNeuralNetwork.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork
{
    internal class TO_BE_DELETEEED
    {
        static void Main(string[] args)
        {

            var trained_nn = INeuralNetwork.Load("2output.nn");
            double[] data = [
                0.50045,
                0.20078,
                0.40056,
                0.871909807,
                0.867575169,
                0.006482074,
                0.222977778, 
                ];
            var result = trained_nn.Predict(data);
            //var denormalized_rez = Math.Exp(result.ElementAt(0)) - 1;
            Console.ReadKey();

        }

        static void da()
        {
            //double[][] inputs = new double[][]
            //{
            //    new double[] { 0.5, 1.0 },
            //    new double[] { -1.0, 2.0 },
            //    new double[] { 0.0, 0.0 },
            //    new double[] { 2.0, -1.0 },
            //    new double[] { 1.5, 2.5 }
            //};

            //double[][] targets = new double[][]
            //{
            //    new double[] { 1.2 },
            //    new double[] { -0.5 },
            //    new double[] { 0.0 },
            //    new double[] { 3.5 },
            //    new double[] { 4.0 }
            //};

            double[][] inputs = new double[][]
            {
                new double[] { 0.1, 0.1 },
                new double[] { 0.1, 0.9 },
                new double[] { 0.9, 0.1 },
                new double[] { 0.9, 0.9 },
            };

            double[][] targets = new double[][]
            {
                new double[] { 0.1 },
                new double[] { 0.9 },
                new double[] { 0.9 },
                new double[] { 0.1 },
            };

            var network = INeuralNetwork.CreateNetwork(3);
            network.AppendLayer(2, LayerType.Input, ActivationFunction.None);
            network.AppendLayer(3, LayerType.Hidden, ActivationFunction.Sigmoid);
            network.AppendLayer(1, LayerType.Output, ActivationFunction.Sigmoid);

            network.Build();
            network.Train(inputs, targets, 100000, 0.1);
        }
    }
}
