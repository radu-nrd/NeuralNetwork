using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NN_MODEL.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_MODEL.Models
{
    public partial class NeuralNetworkModel
    {
        public void SaveModel(string filePath)
        {
            if (Path.GetExtension(filePath).Trim() != ".nn")
                throw new Exception("Only files with extension .nn are allowed");
            BuildJsonFile(filePath);
            var data = File.ReadAllBytes(filePath);
            var encryptionBytes = new byte[data.Length];
            for (int i = 0; i < encryptionBytes.Length; i++)
                encryptionBytes[i] = (byte)(data[i] + 26);
            File.WriteAllText(filePath, Convert.ToBase64String(encryptionBytes));
        }

        public static NeuralNetworkModel LoadModel(string filePath)
        {
            if (Path.GetExtension(filePath).Trim() != ".nn")
                throw new Exception("Only files with extension .nn are allowed");
            var dataString = File.ReadAllText(filePath);
            var data = Convert.FromBase64String(dataString);
            for (int i = 0; i < data.Length; i++)
                data[i] = (byte)(data[i] - 26);
            dataString = Encoding.UTF8.GetString(data);
            JObject obj = JObject.Parse(dataString);
            NeuralNetworkModel model = new NeuralNetworkModel();
            model.LearningRate = Convert.ToDouble(obj["LearningRate"]);
            foreach (var l in (obj["Layers"] as JObject).Values())
            {
                ILayer layer = new Layer(Convert.ToInt32(l["Index"]));
                foreach (var n in l["Neurons"] as JArray)
                {
                    var neuron = new Neuron(layer);
                    neuron.SetBias(Convert.ToDouble(n["Bias"]));
                    layer.AppendNeuron(neuron);
                }
                model.AppendLayer(layer);
            }
            foreach(var kvpInfo in (obj["Weights"] as JObject))
            {
                var weight = new Weight
                {
                    Mapping = kvpInfo.Value["Mapping"].ToString(),
                    Value = Convert.ToDouble(kvpInfo.Value["Value"])
                };
                model._weights.Add(kvpInfo.Key, weight);
            }
            return model;
        }

        private void BuildJsonFile(string filePath)
        {
            using (StreamWriter sw = new StreamWriter(filePath))
            using (JsonTextWriter jsonWriter = new JsonTextWriter(sw))
            {
                jsonWriter.Formatting = Formatting.Indented;

                jsonWriter.WriteStartObject();

                jsonWriter.WritePropertyName("LearningRate");
                jsonWriter.WriteValue(LearningRate);

                jsonWriter.WritePropertyName("Layers");
                jsonWriter.WriteStartObject();
                foreach (var layer in _layers)
                {
                    jsonWriter.WritePropertyName($"Layer{layer.Index}");
                    jsonWriter.WriteStartObject();

                    jsonWriter.WritePropertyName("Index");
                    jsonWriter.WriteValue(layer.Index);
                    jsonWriter.WritePropertyName("Neurons");
                    jsonWriter.WriteStartArray();
                    foreach (var neuron in layer.Neurons)
                    {
                        jsonWriter.WriteStartObject();

                        jsonWriter.WritePropertyName("Tag");
                        jsonWriter.WriteValue(neuron.Tag);

                        jsonWriter.WritePropertyName("Bias");
                        jsonWriter.WriteValue(neuron.Bias);

                        jsonWriter.WriteEndObject();
                    }
                    jsonWriter.WriteEndArray();
                    jsonWriter.WriteEndObject();
                }
                jsonWriter.WriteEndObject();

                jsonWriter.WritePropertyName("Weights");
                jsonWriter.WriteStartObject();
                foreach (var kvp in _weights)
                {
                    jsonWriter.WritePropertyName(kvp.Key);
                    jsonWriter.WriteStartObject();
                    jsonWriter.WritePropertyName("Mapping");
                    jsonWriter.WriteValue(kvp.Value.Mapping);
                    jsonWriter.WritePropertyName("Value");
                    jsonWriter.WriteValue(kvp.Value.Value);
                    jsonWriter.WriteEndObject();
                }
                jsonWriter.WriteEndObject();
                jsonWriter.WriteEndObject();
            }
        }
    }
}
