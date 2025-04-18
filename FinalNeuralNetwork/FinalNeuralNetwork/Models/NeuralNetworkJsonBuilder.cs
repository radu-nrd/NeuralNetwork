using FinalNeuralNetwork.Interfaces;
using FinalNeuralNetwork.Utils;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Models
{
    sealed class NeuralNetworkJsonBuilder
    {
        NeuralNetwork model;

        public NeuralNetworkJsonBuilder(NeuralNetwork model)
        {
            this.model = model;
        }

        public void BuildJsonFile(string filePath)
        {
            using (StreamWriter sw = new StreamWriter(filePath))
            using (JsonTextWriter jsonWriter = new JsonTextWriter(sw))
            {
                jsonWriter.Formatting = Formatting.Indented;

                jsonWriter.WriteStartObject();

                jsonWriter.WritePropertyName("Error");
                jsonWriter.WriteValue(model.Error);

                jsonWriter.WritePropertyName("Layer_Count");
                jsonWriter.WriteValue(model.Layers.Length);

                jsonWriter.WritePropertyName("Layers");
                jsonWriter.WriteStartObject();
                WriteLayers(jsonWriter);
                jsonWriter.WriteEndObject();

                jsonWriter.WritePropertyName("Weights");
                jsonWriter.WriteStartObject();
                WriteWeights(jsonWriter);
                jsonWriter.WriteEndObject();

                jsonWriter.WritePropertyName("Activation_Layer_Functions");
                jsonWriter.WriteStartObject();
                WriteActivationLayerFunctions(jsonWriter);
                jsonWriter.WriteEndObject();

                jsonWriter.WritePropertyName("Layers_type");
                jsonWriter.WriteStartObject();
                WriteLayersType(jsonWriter);
                jsonWriter.WriteEndObject();
            }
        }
        void WriteLayersType(JsonTextWriter jsonTextWriter)
        {
            for(int i = 0;i<model.LayersType.Length;i++)
                WriteLayerType(ref jsonTextWriter, model.LayersType[i],i);
        }
        void WriteLayerType(ref JsonTextWriter jsonTextWriter,LayerType value,int idx) 
        {
            jsonTextWriter.WritePropertyName($"LayerType {idx}");
            jsonTextWriter.WriteStartObject();

            jsonTextWriter.WritePropertyName("Index");
            jsonTextWriter.WriteValue(idx);

            jsonTextWriter.WritePropertyName("Value");
            jsonTextWriter.WriteValue(value.ToString());

            jsonTextWriter.WriteEndObject();
        }

        void WriteActivationLayerFunctions(JsonTextWriter jsonTextWriter)
        {
            for (int i = 0; i < model.ActivationFunctions.Length; i++)
                WriteActivationLayerFunction(ref jsonTextWriter, model.ActivationFunctions[i], i);
        }
        void WriteActivationLayerFunction(ref JsonTextWriter jsonWriter,ActivationFunction value,int idx)
        {
            jsonWriter.WritePropertyName($"Activation {idx}");
            jsonWriter.WriteStartObject();

            jsonWriter.WritePropertyName("Index");
            jsonWriter.WriteValue(idx);

            jsonWriter.WritePropertyName("Value");
            jsonWriter.WriteValue(value.ToString());

            jsonWriter.WriteEndObject();
        }
        void WriteLayers(JsonTextWriter jsonWriter)
        {
            for (int i = 0; i < model.Layers.Length; i++)
                WriteLayer(ref jsonWriter, model.Layers[i], i);
        }
        void WriteLayer(ref JsonTextWriter jsonWriter, double[] layer, int idx)
        {
            jsonWriter.WritePropertyName($"Layer{idx}");
            jsonWriter.WriteStartObject();

            jsonWriter.WritePropertyName("Index");
            jsonWriter.WriteValue(idx);

            jsonWriter.WritePropertyName("Neurons_Biases");
            jsonWriter.WriteStartArray();

            for (int i = 0; i < layer.Length; i++)
                jsonWriter.WriteValue(layer[i]);

            jsonWriter.WriteEndArray();
            jsonWriter.WriteEndObject();
        }
        void WriteWeights(JsonTextWriter jsonWriter)
        {
            for (int w = 1; w < model.Weights.Length; w++)
                WriteWeightsForLayer(ref jsonWriter, model.Weights[w], w);
        }
        void WriteWeightsForLayer(ref JsonTextWriter jsonWriter, double[][] layerWeights, int idx)
        {
            jsonWriter.WritePropertyName($"Weight{idx}");
            jsonWriter.WriteStartObject();

            jsonWriter.WritePropertyName("Index");
            jsonWriter.WriteValue(idx);

            jsonWriter.WritePropertyName("Values");
            jsonWriter.WriteStartObject();
            for(int i = 0;i< layerWeights.Length; i++)
            {
                jsonWriter.WritePropertyName($"Set{i}");
                jsonWriter.WriteStartObject();

                jsonWriter.WritePropertyName("Neuron_Index");
                jsonWriter.WriteValue(i);

                jsonWriter.WritePropertyName("Data");
                jsonWriter.WriteStartArray();
                
                for(int v=0;v< layerWeights[i].Length;v++)
                    jsonWriter.WriteValue(layerWeights[i][v]);
                jsonWriter.WriteEndArray();
                jsonWriter.WriteEndObject();
            }
            jsonWriter.WriteEndObject();
            jsonWriter.WriteEndObject();
        }

    }
}
