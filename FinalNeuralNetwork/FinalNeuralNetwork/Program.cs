// See https://aka.ms/new-console-template for more information
using FinalNeuralNetwork.Interfaces;
using System;
using System.Diagnostics;


//double[][] batch =
//[
//                [0.1,0.1],
//                [0.1,0.9],
//                [0.9,0.1],
//                [0.9,0.9],
//];

//double[][] outcome =
//[
//                //[0.0,1.0],
//                //[1.0,0.0],
//                //[1.0,0.0],
//                //[0.0,1.0],
//                [0.1],
//                [0.9],
//                [0.9],
//                [0.1]
//];

double[][] batch =
{
    new double[] { 0.1, 0.1 }, // Clasa 0
    new double[] { 0.1, 1.0 }, // Clasa 1
    new double[] { 0.9, 0.1 }, // Clasa 2
    new double[] { 0.9, 0.9 }, // Clasa 0
    new double[] { 0.5, 0.9 }, // Clasa 1
    new double[] { 0.9, 0.5 }, // Clasa 2
};

double[][] outcome =
{
    new double[] { 0.9, 0.1,}, // Clasa 0
    new double[] { 0.1, 0.9,}, // Clasa 1
    new double[] { 0.1, 0.1,}, // Clasa 2
    new double[] { 0.9, 0.1,}, // Clasa 0
    new double[] { 0.1, 0.9,}, // Clasa 1
    new double[] { 0.1, 0.1,}, // Clasa 2
};


var nn = INeuralNetwork.CreateNetwork(3);
var inputLayer = new double[2];

nn.AppendLayer(inputLayer);
//nn.AppendLayer(3);
//nn.AppendLayer(1);
nn.AppendLayer(3);
nn.AppendLayer(2,FinalNeuralNetwork.Utils.ActivationFunction.Softmax);

nn.Build();
//nn.Train(batch, outcome,10000);
//var test1 = nn.Predict([0.1, 0.9]);
//nn.Save("new_network_after_GPU_Paralelization.nn");




//var nn = INeuralNetwork.Load("new_network_after_GPU_Paralelization.nn");
nn.Train(batch, outcome, 1000000);
var da = nn.Predict([0.5, 0.9]);
Console.WriteLine("Done");










//var rand = new Random();
//var batch = new double[100_000_000][];
//var outcome = new double[100_000_000][];

//for (int i = 0; i < batch.Length; i++)
//{
//    batch[i] = Enumerable.Range(0, 2).Select(_ => rand.NextDouble()).ToArray();
//    outcome[i] = [rand.NextDouble()];
//}


//var watch = Stopwatch.StartNew();
////nn.Train(batch, outcome, 10000);
//Console.WriteLine($"Done! Time Elapsed: {watch.ElapsedMilliseconds} ms");
//watch.Stop();
//Console.ReadKey();
//nn.TrainGPU(batch, outcome, 1000);
//Console.ReadKey();

