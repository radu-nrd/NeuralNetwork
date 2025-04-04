using FinalNeuralNetwork.Interfaces;
using FinalNeuralNetwork.Utils;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Models
{
    sealed partial class NeuralNetwork
    {
        public static NeuralNetwork Read(string filePath)
        {
            var data = File.ReadAllBytes(filePath);
            for (int i = 0; i < data.Length; i++)
                data[i] = (byte)(data[i] - 26);
            var dataString = Encoding.UTF8.GetString(data);
            JObject jsonObj = JObject.Parse(dataString);
            BuildModel(jsonObj, out var model);
            return model;
        }

        private static void BuildModel(JObject jsonObj, out NeuralNetwork model)
        {
            model = new NeuralNetwork(Convert.ToInt32(jsonObj["Layer_Count"]));
            model.LearningRate = Convert.ToDouble(jsonObj["LearningRate"]);
            model._error = Convert.ToDouble(jsonObj["error"]);

            BuildLayersFromJson(jsonObj, ref model);
            InitializeWeights(ref model);
            BuildWeightsFromJson(jsonObj, ref model);
            BuildActivationLayerFunctionsFromJson(jsonObj,ref model);
        }
        private static void InitializeWeights(ref NeuralNetwork model)
        {
            for(int i = 1; i < model.Weights.Length; i++)
            {
                model._weights[i] = new double[model.Layers[i].Length][];
                for (int j = 0; j < model.Weights[i].Length; j++)
                    model.Weights[i][j] = new double[model.Layers[i - 1].Length];
            }

        }
        private static void BuildActivationLayerFunctionsFromJson(JObject jsonObj, ref NeuralNetwork model)
        {
            var actFcts = jsonObj["Activation_Layer_Functions"]!.Values();
            foreach (var act in actFcts)
            {
                var index = Convert.ToInt32(act["Index"]);
                var data = Enum.Parse<ActivationFunction>(act["Value"]!.ToString());
                model._aFunctions[index] = data;
            }
        }

        private static void BuildWeightsFromJson(JObject jsonObj, ref NeuralNetwork model)
        {
            //model._weights[0] = [Enumerable.Repeat(double.NaN, model._layers[0].Length).ToArray()];
            var weights = jsonObj["Weights"]!.Values();
            foreach(var weightSet in weights)
            {
                var layerIdx = Convert.ToInt32(weightSet["Index"]);
                var sets = weightSet["Values"]!.Values();
                foreach(var set in sets)
                {
                    var neuronIdx = Convert.ToInt32(set["Neuron_Index"]);
                    var data = set["Data"]!.Values().Select(Convert.ToDouble).ToArray();
                    model.Weights[layerIdx][neuronIdx] = data;
                }
            }
        }

        private static void BuildLayersFromJson(JObject jsonObj,ref NeuralNetwork model)
        {
            var layers = jsonObj["Layers"]!.Values(); //not null check!
            foreach (var layer in layers)
            {
                var index = Convert.ToInt32(layer["Index"]);
                var data = layer["Neurons_Biases"]!.Values().Select(Convert.ToDouble).ToArray();
                model._layers[index] = data;
            }
        }
    }
}
