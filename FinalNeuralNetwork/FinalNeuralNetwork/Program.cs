// See https://aka.ms/new-console-template for more information
using FinalNeuralNetwork.Interfaces;
using System;

Console.WriteLine("Hello, World!");

//var nn = INeuralNetwork.CreateNetwork(3);
//var inputLayer = new double[2];
//var hiddenLayer = new double[] { 0.1, 0.3, 0.8 };
//var outputLayer = new double[] { 0.2 };

//nn.AppendLayer(inputLayer);
//nn.AppendLayer(3);
//nn.AppendLayer(1);

//nn.Build();
//nn.Save("new_test_with_XOR.nn");

var nn = INeuralNetwork.Load("new_test_with_XOR.nn");
nn.LearningRate = 0.1;

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
nn.TrainGPU(batch, outcome, 10);







//var watch = new System.Diagnostics.Stopwatch();
//watch.Start();
//nn.Train(batch, outcome, 100000);
//watch.Stop();
//Console.WriteLine();
//Console.WriteLine($"Done CPU test, Time: {watch.Elapsed.TotalSeconds} seconds");
//Console.WriteLine();

//double[] result = [
//    nn.Predict([0.1,0.1]).ElementAt(0),
//     nn.Predict([0.9,0.1]).ElementAt(0),
//      nn.Predict([0.1,0.9]).ElementAt(0),
//       nn.Predict([0.9,0.9]).ElementAt(0),
//    ];

Console.WriteLine("Done");

