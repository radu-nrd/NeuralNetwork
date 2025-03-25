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
            //for (int i = 0; i < data.Length; i++)
            //    data[i] = (byte)(data[i] - 26);
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
            BuildWeightsFromJson(jsonObj, ref model);
            BuildActivationLayerFunctionsFromJson(jsonObj,ref model);
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
            var weights = jsonObj["Weights"]!.Values();
            foreach(var weightSet in weights)
            {
                var index = Convert.ToInt32(weightSet["Index"]);
                var data = weightSet["Values"]!.Values().Select(Convert.ToDouble).ToArray();
                model._weights[index] = data;
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
