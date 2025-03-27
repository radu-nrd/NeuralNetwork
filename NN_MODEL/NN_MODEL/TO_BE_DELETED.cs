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

            //ILayer layer1 = new Layer(1);
            //for (int i = 0; i < 2; i++)
            //    layer1.AppendNeuron(new Neuron(layer1));

            //ILayer layer2 = new Layer(2);
            //for (int i = 0; i < 3; i++)
            //    layer2.AppendNeuron(new Neuron(layer2));

            //ILayer layer3 = new Layer(3);
            //for (int i = 0; i < 1; i++)
            //    layer3.AppendNeuron(new Neuron(layer3));

            //NeuralNetworkModel model = new NeuralNetworkModel();
            //model.AppendLayer(layer1);
            //model.AppendLayer(layer2);
            //model.AppendLayer(layer3);

            //model.BuildNeuralNetwork();
            //model.Setup();

            //model.SaveModel("test_new_merged.nn");

            var model = NeuralNetworkModel.LoadModel("test_new_merged.nn");
            model.Setup();
            var prediction = model.Predict(new double[] { 0.33, 0.21 });

            for(int i = 0;i<10;i++)
                model.Train(new double[] { 0.33, 0.21 }, new double[] { 0.26 });

            prediction = model.Predict(new double[] { 0.33, 0.21 });


            //Random rand = new Random();
            //double[] array = new double[] { 0.33, 0.21 };
            //double[] array = Enumerable.Range(0, 2).Select(_ => rand.NextDouble()).ToArray();

            //var dataFromCpu = model.Predict(array);
            //var dataFromGPU = model.PredictGPU(array);

            //double[][] batch = new double[4][]
            //{
            //    new double[]{0.1,0.1},
            //    new double[]{0.1,0.9},
            //    new double[]{0.9,0.1},
            //    new double[]{0.9,0.9}
            //};

            //double[][] outcome = new double[4][]
            //{
            //    new double[]{0.1},
            //    new double[]{0.9},
            //    new double[]{0.9},
            //    new double[]{0.1},
            //};

            //var watch = new System.Diagnostics.Stopwatch();

            //watch.Start();
            //model.Train(batch, outcome, 1000);
            //watch.Stop();
            //Console.WriteLine();
            //Console.WriteLine($"Done CPU test, Time: {watch.Elapsed.TotalSeconds} seconds");
            //Console.WriteLine();

            //Console.ReadKey();
            //Console.WriteLine("GPU TEST");
            //Console.WriteLine();
            //watch.Reset();
            //watch.Start();
            //model.TrainOnGPU(batch, outcome, 1000);
            //watch.Stop();
            //Console.WriteLine($"Done GPU test, Time: {watch.Elapsed.TotalSeconds} seconds");
            //Console.ReadKey();

            ////var predictAfterTrainCPU = model.Predict(array);
            ////var predictAfterTrainGPU = model.PredictGPU(array);

            //Console.ReadKey();
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
