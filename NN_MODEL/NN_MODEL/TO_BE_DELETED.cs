using NN_MODEL.Interfaces;
using NN_MODEL.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_MODEL
{
    internal class TO_BE_DELETED
    {
        static void Main(string[] args)
        {
            ILayer layer1 = new Layer(1);
            layer1.AppendNeurons(new INeuron[] { new Neuron(layer1), new Neuron(layer1) });

            ILayer layer2 = new Layer(2);
            layer2.AppendNeurons(new INeuron[] { new Neuron(layer2), new Neuron(layer2), new Neuron(layer2) });

            ILayer layer3 = new Layer(3);
            layer3.AppendNeurons(new INeuron[] { new Neuron(layer3) , new Neuron(layer3)});

            NeuralNetworkModel model = new NeuralNetworkModel();
            model.AppendLayer(layer1);
            model.AppendLayer(layer2);
            model.AppendLayer(layer3);

            model.Build();
            //model.Predict(new double[] { 0.33, 0.66 });
            for (int i = 0; i < 1000; i++)
            {
                Console.WriteLine($"Iteration {i}");
                model.Train(new double[] { 0.33,0.66 }, new double[] { 0.9, 0.1 });
                Console.WriteLine();
            }
            var prediction = model.Predict(new double[] { 0.33, 0.66 });
            //model.Train(new double[] { 0.33, 0.66 }, new double[] { 0.75, 0.25 });
            //model.Train(new double[] { 0.33, 0.66 }, new double[] { 0.75, 0.25 });
            //model.Train(new double[] { 0.33, 0.66 }, new double[] { 0.75, 0.25 });
            //model.Train(new double[] { 0.33, 0.66 }, new double[] { 0.75, 0.25 });

        }
    }
}
