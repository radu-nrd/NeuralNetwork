// See https://aka.ms/new-console-template for more information
using FinalNeuralNetwork.Interfaces;
using ILGPU.Runtime.Cuda;

Console.WriteLine("Hello, World!");

//var nn = INeuralNetwork.CreateNetwork(3);
//var inputLayer = new double[2];
//var hiddenLayer = new double[] { 0.1, 0.3, 0.8 };
//var outputLayer = new double[] { 0.2 };

//nn.AppendLayer(inputLayer);
//nn.AppendLayer(hiddenLayer);
//nn.AppendLayer(outputLayer);

//nn.Build();
//nn.Save("test_nn_new_format.nn");

var nn = INeuralNetwork.Load("test_nn_new_format.nn");
var prediction = nn.Predict([0.33, 0.21]);

for (int i = 0; i < 10; i++)
    nn._Train([0.33, 0.21], [0.26]);

prediction = nn.Predict([0.33, 0.21]);
Console.WriteLine("Done");

    