using NN_MODEL.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NN_MODEL.Models
{
    public enum Direction
    {
        Forward,
        Backward,
    }

    public partial class NeuralNetworkModel
    {
        readonly List<ILayer> _layers;
        public IReadOnlyCollection<ILayer> Layers => _layers;

        public NeuralNetworkModel()
        {
            _layers = new List<ILayer>();
            _index = 1;
            _weights = new Dictionary<string, Weight>();
            _regex = new Regex(@"W\$(N\d+)\$(N\d+)");
            LearningRate = 0.1;
        }
        public void AppendLayer(ILayer layer)
        {
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
            //return ApplySoftmax(_tmpData);
            return ApplySigmoid(_tmpData);

        }
        private IEnumerable<double> ApplySigmoid(IEnumerable<double> data)
        {
            var dataCache = new List<double>();
            foreach (var elem in data)
                dataCache.Add(1 / (1 + Math.Exp(-elem)));
            return dataCache;
        }
        private double ApplySigmoid(double data)
        {
            return 1 / (1 + Math.Exp(-data));
        }
        private double ApplySigmoidDerivative(double data)
        {
            return data * (1 - data);
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
                sum += Math.Pow(prediction.ElementAt(i) - outcome.ElementAt(i), 2);
            return sum;
        }
        public void Train(double[][] batch, double[][] outcome,int epochs)
        {
            Console.WriteLine("Starting Train!");
            //foreach(var key in _weights.Keys)
            //{
            //    Console.WriteLine($"{key} : {_weights[key].Value}");
            //}
            //Console.WriteLine();
            for (int e = 0;e<epochs;e++)
            {
                double totalError = 0.0;
                for (int i = 0; i < batch.Length; i++)
                {
                    var data = batch[i];
                    var valid_prediction = outcome[i];
                    var network_prediction = Predict(data);
                    totalError += MSE(network_prediction,valid_prediction);
                    Train(data, valid_prediction);
                }
                Console.WriteLine($"Epoch {e + 1}/{epochs}: Total Error: {totalError}");
                //foreach (var key in _weights.Keys)
                //{
                //    Console.WriteLine($"{key} : {_weights[key].Value}");
                //}
                //Console.WriteLine();
            }
        }

        public void Train(IEnumerable<double> input, IEnumerable<double> outcome)
        {
            IEnumerable<double> prediction;
            var _tmpForwardSave = GetDataFromForward(input,out prediction);

            #region Backpropagation
            var outputError = new List<double>();
            for (int i = 0; i < outcome.Count(); i++)
                outputError.Add(outcome.ElementAt(i) - prediction.ElementAt(i));

            var outputGradient = CalculateGradient(outputError, prediction);
            var lastGradient = outputGradient;

            for (int i = 0; i < _layers.Last().Neurons.Count; i++)
            {
                var n = _layers.Last().Neurons.ElementAt(i);
                n.AdjustBias(LearningRate * lastGradient.ElementAt(i));
            }

            for (int i = _layers.Count - 2; i > 0; i--)
            {
                IEnumerable<double> inputFromBackward;
                UpdateWeights(_layers.ElementAt(i), lastGradient, _tmpForwardSave[_layers.ElementAt(i).Index]);
                inputFromBackward = Backward(_layers.ElementAt(i), lastGradient);
                lastGradient = CalculateGradient(inputFromBackward, _tmpForwardSave[_layers.ElementAt(i).Index]);
                UpdateBiases(_layers.ElementAt(i), lastGradient);
            }
            UpdateWeights(_layers.First(), lastGradient, _tmpForwardSave[_layers.First().Index]);
            #endregion
        }

        private Dictionary<int,IEnumerable<double>> GetDataFromForward(IEnumerable<double> input, out IEnumerable<double> prediction)
        {
            Dictionary<int, IEnumerable<double>> _tmpForwardSave = new Dictionary<int, IEnumerable<double>>();
            var _tmpData = input;
            _tmpForwardSave.Add(_layers.First().Index, input);
            for (int i = 1; i < _layers.Count - 1; i++)
            {
                _tmpData = Forward(_layers[i], _tmpData);
                _tmpData = ApplySigmoid(_tmpData);
                _tmpForwardSave.Add(_layers[i].Index, _tmpData);
            }
            _tmpData = Forward(_layers.Last(), _tmpData);
            prediction = ApplySigmoid(_tmpData);
            _tmpForwardSave.Add(_layers.Last().Index, prediction);

            return _tmpForwardSave;
        }

        private void UpdateBiases(ILayer layer, IEnumerable<double> gradient)
        {
            if (layer.Neurons.Count != gradient.Count())
                throw new ArgumentException("Neurons count and gradient count doesn't match");
            for(int i = 0; i < layer.Neurons.Count; i++)
            {
                var n = layer.Neurons.ElementAt(i);
                n.AdjustBias(LearningRate * gradient.ElementAt(i));
            }
        }
        private void UpdateWeights(ILayer layer,IEnumerable<double> gradient,IEnumerable<double> forwardData)
        {
            //if (gradient.Count() != layer.Neurons.Count())
                //throw new ArgumentException("The gradient count don't match the layer neuron count");

            for(int i = 0;i<layer.Neurons.Count();i++)
            {
                var n = layer.Neurons.ElementAt(i);
                var weightsKeys = this.GetWeightKeys(n, Direction.Backward);
                if (weightsKeys.Count() != gradient.Count())
                    throw new ArgumentException("The gradient count don't match the weights count for backward!");
                for (int j = 0; j < weightsKeys.Count(); j++)
                    _weights[weightsKeys.ElementAt(j)].Value += LearningRate * gradient.ElementAt(j)*forwardData.ElementAt(i);
            }

        }

        private IEnumerable<double> CalculateGradient(IEnumerable<double> inputFromBackward,IEnumerable<double>dataFromForward)
        {
            var _tmpGradient = new double[inputFromBackward.Count()];
            for(int i=0; i<inputFromBackward.Count(); i++)
                _tmpGradient[i] = inputFromBackward.ElementAt(i) * ApplySigmoidDerivative(dataFromForward.ElementAt(i));
            return _tmpGradient.AsEnumerable();
        }

        private IEnumerable<double> SoftmaxDerivative(IEnumerable<double> input,IEnumerable<double> outcome)
        {
            var dataCache = new List<double>();
            if (input.Count() != outcome.Count())
                throw new ArgumentException("Input and outcome is not the same size!");

            for(int i=0;i<input.Count();i++)
                dataCache.Add(input.ElementAt(i)-outcome.ElementAt(i));
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
            var weightValues = this.GetWeightValues(neuron, Direction.Forward);
            var rez = double.NaN;
            if (data.Count() == weightValues.Count())
            {
                rez = neuron.Bias;
                for (int i = 0; i < data.Count(); i++)
                    rez += weightValues.ElementAt(i) * data.ElementAt(i);
            }
            if (double.IsNaN(rez))
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
            var weightValues = this.GetWeightValues(neuron,Direction.Backward);
            var rez = double.NaN;
            if(data.Count() == weightValues.Count())
            {
                rez = 0;
                for (int i = 0; i < data.Count(); i++)
                    rez += weightValues.ElementAt(i) * data.ElementAt(i);
            }
            if (double.IsNaN(rez))
                throw new ArgumentException("The given data cannot be backward!");
            return rez;
        }
    }
}
