using Newtonsoft.Json;
using NN_MODEL.Interfaces;
using System;
using System.CodeDom;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NN_MODEL.Models
{
    public partial class NeuralNetworkModel
    {
        readonly Dictionary<string, Weight> _weights;
        private int _index;
        readonly Regex _regex;
        public double LearningRate { get; set; }
        private void InitializeWeightsValues()
        {
            foreach (var key in _weights.Keys)
                _weights[key].Value = (Utility.Rand.NextDouble() * 2) - 1;
        }
        public void BuildNeuralNetwork()
        {
            for (int i = 0; i < Layers.Count - 1; i++)
                BuildWeightMapEntry(Layers.ElementAt(i), Layers.ElementAt(i + 1));
            InitializeWeightsValues();
        }

        public IEnumerable<string> GetWeightKeys(INeuron neuron,Direction direction)
        {
            var cache = new List<string>();

            if(direction == Direction.Forward)
                foreach(var e in _weights)
                {
                    var conWeight = _regex.Match(e.Value.Mapping).Groups[2].Value;
                    if (neuron.Tag == conWeight)
                        cache.Add(e.Key);
                }
            if(direction == Direction.Backward)
                foreach(var e in _weights)
                {
                    var conWeight = _regex.Match(e.Value.Mapping).Groups[1].Value;
                    if (neuron.Tag == conWeight)
                        cache.Add(e.Key);
                }
            return cache;
        }
        public IEnumerable<double> GetWeightValues(INeuron neuron,Direction direction)
        {
            var cache = new List<double>();
            var keys = GetWeightKeys(neuron,direction);
            foreach (var key in keys)
                cache.Add(_weights[key].Value);
            return cache;
        }
        private void BuildWeightMapEntry(ILayer l1,ILayer l2)
        {
            foreach(var n1 in l1.Neurons)
                foreach(var n2 in l2.Neurons)
                    BuildWeightMapEntry(n1, n2);
        }
        private void BuildWeightMapEntry(INeuron n1, INeuron n2)
        {
            var key = $"W{_index}";
            var value = new Weight { Mapping = $"W${n1.Tag}${n2.Tag}" };
            _weights.Add(key, value);
            _index++;
        }
       
    }
}
