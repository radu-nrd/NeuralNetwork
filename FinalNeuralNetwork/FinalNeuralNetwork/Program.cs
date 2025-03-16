// See https://aka.ms/new-console-template for more information
using FinalNeuralNetwork.Interfaces;

Console.WriteLine("Hello, World!");

var nn = INeuralNetwork.CreateNetwork(3);
var inputLayer = new double[] { 0.3, 0.1, 0.2 };

nn.AppendLayer(inputLayer);


