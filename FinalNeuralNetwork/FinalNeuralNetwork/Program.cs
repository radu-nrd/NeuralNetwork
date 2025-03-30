// See https://aka.ms/new-console-template for more information
using FinalNeuralNetwork.Interfaces;
using System;

Console.WriteLine("Hello, World!");

var nn = INeuralNetwork.CreateNetwork(3);
var inputLayer = new double[2];
var hiddenLayer = new double[] { 0.1, 0.3, 0.8 };
var outputLayer = new double[] { 0.2 };

nn.AppendLayer(inputLayer);
nn.AppendLayer(3);
nn.AppendLayer(1);

nn.Build();
nn.Save("new_test_with_XOR.nn");

//var nn = INeuralNetwork.Load("new_test_with_XOR.nn");
//var random = new Random();

//var batch = new double[5][]
//{
//                Enumerable.Range(0,16384).Select(_=>random.NextDouble()).ToArray(),
//                Enumerable.Range(0,16384).Select(_=>random.NextDouble()).ToArray(),
//                Enumerable.Range(0,16384).Select(_=>random.NextDouble()).ToArray(),
//                Enumerable.Range(0,16384).Select(_=>random.NextDouble()).ToArray(),
//                Enumerable.Range(0,16384).Select(_=>random.NextDouble()).ToArray()
//};


//double[][] outcome =
//[
//                [0.123],
//                [0.123],
//                [0.123],
//                [0.123],
//                [0.123]
//];

double[][] batch =
[
                [0.1,0.1],
                [0.1,0.9],
                [0.9,0.1],
                [0.9,0.9]
];

double[][] outcome =
[
                [0.1],
                [0.9],
                [0.9],
                [0.1],
];

var watch = new System.Diagnostics.Stopwatch();
watch.Start();
nn.Train(batch, outcome, 1000);
watch.Stop();
Console.WriteLine();
Console.WriteLine($"Done CPU test, Time: {watch.Elapsed.TotalSeconds} seconds");
Console.WriteLine();
Console.WriteLine("Done");

