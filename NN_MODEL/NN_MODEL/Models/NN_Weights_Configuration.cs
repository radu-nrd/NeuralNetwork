using NN_MODEL.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NN_MODEL.Models
{
    class NN_Weights_Configuration
    {
        readonly Dictionary<string, Weight> _weights;
        private int __index;
        readonly NeuralNetworkModel _main;
        readonly Regex _regex;
        public NN_Weights_Configuration(NeuralNetworkModel main)
        {
            __index = 1;
            _weights = new Dictionary<string, Weight>();
            _main = main;
            _regex = new Regex(@"W\$(N\d+)\$(N\d+)");
        }
        public double this[string key] {
            get
            {
                if (_weights.ContainsKey(key))
                    return _weights[key].Value;
                throw new ArgumentException("Invalid key");
            }
            set
            {
                _weights[key].Value = value;
            }
        }
        private void InitializeWeightsValues()
        {
            foreach (var key in _weights.Keys)
                _weights[key].Value = Utility.Rand.NextDouble();
        }
        public void BuildNeuralNetwork()
        {
            for (int i = 0; i < _main.Layers.Count - 1; i++)
                BuildWeightMapEntry(_main.Layers.ElementAt(i), _main.Layers.ElementAt(i + 1));
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
            var key = $"W{__index}";
            var value = new Weight { Mapping = $"W${n1.Tag}${n2.Tag}" };
            _weights.Add(key, value);
            __index++;
        }


    }
}
