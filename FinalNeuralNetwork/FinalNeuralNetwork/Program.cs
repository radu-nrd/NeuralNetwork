// See https://aka.ms/new-console-template for more information
using FinalNeuralNetwork.Interfaces;
using System;
using System.Diagnostics;


double[][] batch =
[
                [0.1,0.1],
                [0.1,0.9],
                [0.9,0.1],
                [0.9,0.9],
];

double[][] outcome =
[
                [0.0,0.0,1.0],
                [1.0,0.0,0.0],
                [1.0,0.0,0.0],
                [0.0,0.0,1.0],
];


var nn = INeuralNetwork.CreateNetwork(3);
var inputLayer = new double[2];
var hiddenLayer = new double[] { 0.1, 0.3, 0.8 };
var outputLayer = new double[] { 0.2 };

nn.AppendLayer(inputLayer);
nn.AppendLayer(8,FinalNeuralNetwork.Utils.ActivationFunction.Relu);
nn.AppendLayer(3,FinalNeuralNetwork.Utils.ActivationFunction.Softmax);

nn.Build();
nn.Train(batch, outcome,10000000);
var test1 = nn.Predict([0.1, 0.9]);
nn.Save("new_network_after_GPU_Paralelization.nn");




//var nn = INeuralNetwork.Load("new_test_with_XOR.nn");
//nn.LearningRate = 0.1;



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

