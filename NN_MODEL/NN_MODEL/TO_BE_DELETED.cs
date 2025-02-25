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
            layer2.AppendNeurons(new INeuron[] { new Neuron(layer2), new Neuron(layer2), new Neuron(layer2)});

            ILayer layer3 = new Layer(3);
            layer3.AppendNeurons(new INeuron[] { new Neuron(layer3), new Neuron(layer3), new Neuron(layer3) });

            ILayer layer4 = new Layer(4);
            layer4.AppendNeurons(new INeuron[] { new Neuron(layer4), new Neuron(layer4), new Neuron(layer4) });

            ILayer layer5 = new Layer(5);
            layer5.AppendNeurons(new INeuron[] { new Neuron(layer5) });

            NeuralNetworkModel model = new NeuralNetworkModel();
            model.AppendLayer(layer1);
            model.AppendLayer(layer2);
            model.AppendLayer(layer3);
            model.AppendLayer(layer4);
            model.AppendLayer(layer5);

            model.Build();
            //model._config["W1"] = 0.2;
            //model._config["W2"] = -0.3;
            //model._config["W3"] = -0.5;
            //model._config["W4"] = 0.1;
            //model._config["W5"] = 0.4;
            //model._config["W6"] = -0.2;
            //model._config["W7"] = 0.3;
            //model._config["W8"] = -0.7;
            //model._config["W9"] = 0.5;

            //model.Train(new double[] { 0.5, -0.4 }, new double[] { 0.6 });


            //var value = model.Predict(new double[] { 0.5, -0.4 });

            //double[][] inputs = { new double[] { 0.1, 0.1 }, new double[] { 0.1, 0.9 } }; //new double[] { 0.9, 0.1 }, new double[] { 0.9, 0.9 } };
            //double[][] outputs = { new double[] { 0.1 }, new double[] { 0.9 } }; //new double[] { 0.9 }, new double[] { 0.1 } };

            for (int i = 0; i < 1000000; i++)
            {
                Console.WriteLine($"Iteration {i}");
                //model.Train(inputs, outputs);
                model.Train(new double[] { 0.1, 0.1}, new double[] { 0.1 });
                model.Train(new double[] { 0.9, 0.9 }, new double[] { 0.1 });
                model.Train(new double[] { 0.1, 0.9 }, new double[] {0.9 });
                model.Train(new double[] { 0.9, 0.1 }, new double[] { 0.9 });
                Console.WriteLine();
            }


            var da = model.Predict(new double[] { 0.1, 0.9 });
            //for (int i = 0; i < 1000; i++)
            //{
            //    Console.WriteLine($"Iteration {i}");
            //    model.Train(new double[] { 0.11, 0.26 }, new double[] { 0.22, 0.78 });
            //    Console.WriteLine();
            //}
            //var prediction = model.Predict(new double[] { 0.11, 0.26 });
            //model.Train(new double[] { 0.33, 0.66 }, new double[] { 0.75, 0.25 });
            //model.Train(new double[] { 0.33, 0.66 }, new double[] { 0.75, 0.25 });
            //model.Train(new double[] { 0.33, 0.66 }, new double[] { 0.75, 0.25 });
            //model.Train(new double[] { 0.33, 0.66 }, new double[] { 0.75, 0.25 });

        }
    }
}
