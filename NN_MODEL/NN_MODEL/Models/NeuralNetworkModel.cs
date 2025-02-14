using NN_MODEL.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NN_MODEL.Models
{
    public enum Direction
    {
        Forward,
        Backward,
    }

    public class NeuralNetworkModel
    {
        readonly List<ILayer> _layers;
        readonly NN_Weights_Configuration _config;

        public IReadOnlyCollection<ILayer> Layers => _layers;

        public NeuralNetworkModel()
        {
            _layers = new List<ILayer>();
            _config = new NN_Weights_Configuration(this);
        }
        public void Build()
        {
            _config.BuildNeuralNetwork();
        }
        public void AppendLayer(ILayer layer)
        {
            //if (_layers.Any())
            //{
            //    var lastLayer = _layers.Last();
            //    foreach(var n1 in  lastLayer.Neurons)
            //        foreach(var n2 in layer.Neurons)
            //            BuildWeightMapEntry(n1, n2);
            //}
            _layers.Add(layer);
        }
        public IEnumerable<double> Predict(IEnumerable<double> input)
        {
            IEnumerable<double> _tmpData;
            _tmpData = input;
            for(int i = 1; i < _layers.Count-1; i++)
            {
                _tmpData = Forward(_layers[i], _tmpData);
                _tmpData = ApplySigmoid(_tmpData);
            }
            _tmpData = Forward(_layers.Last(), _tmpData);
            return ApplySoftmax(_tmpData);

        }
        private IEnumerable<double> ApplySigmoid(IEnumerable<double> data)
        {
            var dataCache = new List<double>();
            foreach (var elem in data)
                dataCache.Add(1 / (1 + Math.Exp(-elem)));
            return dataCache;
        }
        private double ApplySigmoidDerivative(double data)
        {
            return data * (1-data);
        }
        private IEnumerable<double> ApplySoftmax(IEnumerable<double> data)
        {
            var maxValue = data.Max();
            var expValues = data.Select(v=>Math.Exp(v - maxValue));
            var sumExp = expValues.Sum();
            return expValues.Select(v => v / sumExp);
            
        }
        private double MSE(IEnumerable<double> prediction, IEnumerable<double> outcome)
        {
            double sum = 0.0;
            for(int i = 0; i < prediction.Count(); i++)
                sum += Math.Pow(outcome.ElementAt(i) - prediction.ElementAt(i), 2);
            return sum;
        }
        public void Train(IEnumerable<double> input,IEnumerable<double> outcome)
        {
            var prediction = Predict(input);
            var loss = MSE(prediction, outcome);
            Console.WriteLine($"MSE: {loss}");
            var outputGradient = SoftmaxDerivative(prediction, outcome);
            var lastGradient = outputGradient;

            for(int i=_layers.Count-2; i >= 0; i--)
            {
                IEnumerable<double> error;
                error = Backward(_layers.ElementAt(i), lastGradient);
                UpdateNetwork(_layers.ElementAt(i), lastGradient);
                lastGradient = CalculateGradient(error);
            }
        }
        private void UpdateNetwork(ILayer layer,IEnumerable<double> gradient)
        {
            //if (gradient.Count() != layer.Neurons.Count())
                //throw new ArgumentException("The gradient count don't match the layer neuron count");

            for(int i = 0;i<layer.Neurons.Count();i++)
            {
                var n = layer.Neurons.ElementAt(i);
                var weightsKeys = _config.GetWeightKeys(n, Direction.Backward);
                if (weightsKeys.Count() != gradient.Count())
                    throw new ArgumentException("The gradient count don't match the weights count for backward!");
                var biasGradient = 0.0;
                for (int j = 0; j < weightsKeys.Count(); j++)
                {
                    biasGradient += gradient.ElementAt(j) * _config[weightsKeys.ElementAt(j)];
                    _config[weightsKeys.ElementAt(j)] -= 0.01 * gradient.ElementAt(j);

                }
                n.AdjustBias(0.01 * biasGradient);
            }

        }
        private IEnumerable<double> CalculateGradient(IEnumerable<double> error)
        {
            var _tmpGradient = new double[error.Count()];
            for(int i=0; i<error.Count(); i++)
                _tmpGradient[i] = error.ElementAt(i) * ApplySigmoidDerivative(error.ElementAt(i));
            return _tmpGradient.AsEnumerable();
        }
        private IEnumerable<double> SoftmaxDerivative(IEnumerable<double> input,IEnumerable<double> outcome)
        {
            var dataCache = new List<double>();
            if (input.Count() != outcome.Count())
                throw new ArgumentException("Input and outcome is not the same size!");

            for(int i=0;i<input.Count();i++)
                dataCache.Add(input.ElementAt(i) - outcome.ElementAt(i));
            return dataCache;
        }

        private IEnumerable<double> Forward(ILayer layer,IEnumerable<double> data)
        {
            var dataCache = new List<double>();
            foreach(var n in layer.Neurons)
                dataCache.Add(Forward(n,data));
            return dataCache;
        }

        private double Forward(INeuron neuron,IEnumerable<double>data)
        {
            var weightValues = _config.GetWeightValues(neuron, Direction.Forward);
            var rez = double.NaN;
            if (data.Count() == weightValues.Count())
            {
                rez = neuron.Bias;
                for (int i = 0; i < data.Count(); i++)
                    rez += weightValues.ElementAt(i) * data.ElementAt(i);
            }
            if (rez == double.NaN)
                throw new ArgumentException("The given data cannot be forward!");
            return rez;
        }

        private IEnumerable<double> Backward(ILayer layer, IEnumerable<double> data)
        {
            var dataCache = new List<double>();
            foreach(var n in layer.Neurons)
                dataCache.Add(Backward(n,data));
            return dataCache;
        }

        private double Backward(INeuron neuron, IEnumerable<double> data)
        {
            var weightValues = _config.GetWeightValues(neuron,Direction.Backward);
            var rez = double.NaN;
            if(data.Count() == weightValues.Count())
            {
                rez = 0;
                for (int i = 0; i < data.Count(); i++)
                    rez += weightValues.ElementAt(i) * data.ElementAt(i);
            }
            if (rez == double.NaN)
                throw new ArgumentException("The given data cannot be backward!");
            return rez;
        }
    }
}
