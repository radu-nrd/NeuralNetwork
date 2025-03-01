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

            var model = NeuralNetworkModel.LoadModel("model.nn");
            double[][] inputs = { new double[] { 0.1, 0.1 }, new double[] { 0.1, 0.9 }, new double[] { 0.9, 0.1 }, new double[] { 0.9, 0.9 } };
            double[][] outputs = { new double[] { 0.1 }, new double[] { 0.9 }, new double[] { 0.9 }, new double[] { 0.1 } };


            var da1 = model.Predict(new double[] { 0.1, 0.9 });
            var da2 = model.Predict(new double[] { 0.9, 0.9 });
            var da3 = model.Predict(new double[] { 0.9, 0.1 });
            var da4 = model.Predict(new double[] { 0.1, 0.1 });
        }

        static void SaveBuiltNeuralNetwork()
        {
            //ILayer layer1 = new Layer(1);
            //layer1.AppendNeurons(new INeuron[] { new Neuron(layer1), new Neuron(layer1) });

            //ILayer layer2 = new Layer(2);
            //layer2.AppendNeurons(new INeuron[] { new Neuron(layer2), new Neuron(layer2), new Neuron(layer2) });

            ////ILayer layer3 = new Layer(3);
            ////layer3.AppendNeurons(new INeuron[] { new Neuron(layer3), new Neuron(layer3), new Neuron(layer3) });

            ////ILayer layer4 = new Layer(4);
            ////layer4.AppendNeurons(new INeuron[] { new Neuron(layer4), new Neuron(layer4), new Neuron(layer4) });

            //ILayer layer3 = new Layer(3);
            //layer3.AppendNeurons(new INeuron[] { new Neuron(layer3) });

            //NeuralNetworkModel model = new NeuralNetworkModel();
            //model.AppendLayer(layer1);
            //model.AppendLayer(layer2);
            //model.AppendLayer(layer3);
            ////model.AppendLayer(layer4);
            ////model.AppendLayer(layer5);

            //model.BuildNeuralNetwork();




            //double[][] inputs = { new double[] { 0.1, 0.1 }, new double[] { 0.1, 0.9 }, new double[] { 0.9, 0.1 }, new double[] { 0.9, 0.9 } };
            //double[][] outputs = { new double[] { 0.1 }, new double[] { 0.9 }, new double[] { 0.9 }, new double[] { 0.1 } };
            //model.Train(inputs, outputs, 100000);
            //model.SaveModel("model.nn");


            //var da1 = model.Predict(new double[] { 0.1, 0.9 });
            //var da2 = model.Predict(new double[] { 0.9, 0.9 });
            //var da3 = model.Predict(new double[] { 0.9, 0.1 });
            //var da4 = model.Predict(new double[] { 0.1, 0.1 });
        }
    }
}
