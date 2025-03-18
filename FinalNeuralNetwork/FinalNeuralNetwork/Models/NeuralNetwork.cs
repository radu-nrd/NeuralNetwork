using FinalNeuralNetwork.Exceptions;
using FinalNeuralNetwork.Interfaces;
using FinalNeuralNetwork.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Models
{   
    sealed class NeuralNetwork : INeuralNetwork
    {

        private readonly double[][] _layers;
        private readonly double[][] _weights;
        private readonly ActivationFunction[] _aFunctions;

        private double _lr;
        private bool _isBuilt;
        private double _error;

        public double LearningRate { get => _lr; set { _lr = value; } }
        public double Error => _error;
        public bool IsBuilt => _isBuilt;

        
        public NeuralNetwork(int layersCount)
        {
            _layers = new double[layersCount][];
            _weights = new double[layersCount][];
            _aFunctions = new ActivationFunction[layersCount];
        }
        private void GetFreeLayerPosition(out int idx)
        {
            idx = -1;
            for (int i = 0; i < _layers.Length; i++)
                if (_layers[i] is null)
                {
                    idx = i;
                    return;
                }
        }

        private void RandomInitializeLayer(ref double[] layer)
        {
            for(int i=0;i<layer.Length; i++)
                layer[i] = Utils.Utils.Random.NextDouble();
        }
        public void AppendLayer(int neuronsCount)
        {
            var layer = new double[neuronsCount];
            RandomInitializeLayer(ref layer);
            this.AppendLayer(layer);
        }
        public void AppendLayer(int neuronsCount, ActivationFunction actv)
        {
            var layer = new double[neuronsCount];
            RandomInitializeLayer(ref layer);
            this.AppendLayer(layer,actv);
        }
        public void AppendLayer(double[] layer)
        {
            this.AppendLayer(layer, ActivationFunction.Sigmoid);
        }

        public void AppendLayer(IEnumerable<double> layer)
        {
            this.AppendLayer(layer.ToArray(),ActivationFunction.Sigmoid);
        }

        public void AppendLayer(double[] layer, ActivationFunction actv)
        {
            GetFreeLayerPosition(out var idx);

            if (idx == 0)
                if (!IsEmptyVector(layer))
                    throw new InvalidAppendLayerException("Cannot append a input layer that is not empty!");

            if (idx == -1)
                throw new InvalidAppendLayerException("Neural network is at maximum size!");

            _layers[idx] = layer;
            _aFunctions[idx] = actv;
        }
        private bool IsEmptyVector(double[] vector) => !vector.Any(x=>x!=0);
        private bool CheckIfIsReadyForBuild() => !_layers.Any(l => l is null);
        public void AppendLayer(IEnumerable<double> layer, ActivationFunction actv)
        {
            this.AppendLayer(layer.ToArray(), actv);
        }

        public void Build()
        {
            if (!CheckIfIsReadyForBuild())
                throw new NotReadyForBuildException("Failed to build network. One or more layers are null");

            for(int i = 1;i< _layers.Length; i++)
            {
                var currentLayer = _layers[i];
                var prevLayer = _layers[i - 1];
                _weights[i] = new double[prevLayer.Length * currentLayer.Length];
            }
        }

        public IReadOnlyCollection<double> Predict(double[] input)
        {
            throw new NotImplementedException();
        }

        public IReadOnlyCollection<double> Predict(IEnumerable<double> input)
        {
            throw new NotImplementedException();
        }

        public void Train(double[][] batch, double[][] validResult, int epochs)
        {
            throw new NotImplementedException();
        }

        public void Train(double[,] batch, double[,] validResult, int epochs)
        {
            throw new NotImplementedException();
        }

        public void Train(IEnumerable<IEnumerable<double>> batch, IEnumerable<IEnumerable<double>> validResult, int epochs)
        {
            throw new NotImplementedException();
        }

        private double[] ExtractDataFromArray(int startIndex,int count,double[] data)
        {
            double[] result = new double[startIndex+count];
            for(int i = startIndex; i< startIndex+count; i++)
                result[i] = data[i];
            return result;
        }

        private Dictionary<int, double[]> GetDataFromForward(double[] input)
        {
            var _tmpForwardSave = new Dictionary<int, double[]>();
            var _tmpData = input;
            _tmpForwardSave.Add(0, input);
            
            for(int i = 1; i < _layers.Length; i++)
            {
                var biases = _layers[i];
                var weights = _weights[i];
                var forwardData = ForwardThroughNetwork(_tmpData, biases, weights);
            }
            return _tmpForwardSave;
        }
        private double[] ForwardThroughNetwork(double[] input,double[] layer, double[] weights)
        {
            double[] result = new double[layer.Length];
            for(int n=0;n<layer.Length; n++)
            {
                var sum = layer[n];
                var startIndex = n * input.Length;
                var count = input.Length;
                var pWeights = ExtractDataFromArray(startIndex, count, weights);

                for (int i = 0; i < input.Length; i++)
                    sum += input[i] * pWeights[i];
                result[n] = sum;
            }
            return result;
        }

        
    }
}
