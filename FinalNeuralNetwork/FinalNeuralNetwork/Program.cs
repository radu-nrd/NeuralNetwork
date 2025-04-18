// See https://aka.ms/new-console-template for more information
using FinalNeuralNetwork.Interfaces;
using FinalNeuralNetwork.Utils;
using System;
using System.Diagnostics;

double[][] batch =
{
    new double[] { 0.1, 0.1 },
    new double[] { 0.9, 0.1 },
    new double[] { 0.1, 0.9 },
    new double[] { 0.9, 0.9 },
};

double[][] outcome =
{
    new double[] { 0.1,0.9 },
    new double[] { 0.9,0.1 },
    new double[] { 0.9,0.1},
    new double[] { 0.1,0.9},
};

//double[][] batch =
//{
//    new double[] { 0, 0 },
//    new double[] { 1, 0 },
//    new double[] { 0, 1 },
//    new double[] { 1, 1},
//};

//double[][] outcome =
//{
//    new double[] { 0,1},
//    new double[] { 1,0},
//    new double[] { 1,0},
//    new double[] { 0,1},
//};

var model = INeuralNetwork.CreateNetwork(3);
model.AppendLayer(2, LayerType.Input);
model.AppendLayer(8, LayerType.Hidden,ActivationFunction.Relu);
model.AppendLayer(2, LayerType.Output,ActivationFunction.Relu);

model.Build();
model.Train(batch, outcome, 10000,0.01);
var pred = model.Predict([1, 1]); // 0,1

Console.WriteLine("Done!!!");

