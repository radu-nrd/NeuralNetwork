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
    sealed partial class NeuralNetwork : INeuralNetwork
    {

        private readonly double[][] _layers;
        private readonly double[][][] _weights;
        private readonly ActivationFunction[] _aFunctions;

        private bool _isBuilt;
        private double _error;

        public bool IsBuilt => _isBuilt;

        public double[][][] Weights => _weights;

        public double[][] Layers => _layers;

        public ActivationFunction[] ActivationFunctions => _aFunctions;

        public double Error => _error;

        public NeuralNetwork(int layersCount)
        {
            _layers = new double[layersCount][];
            _weights = new double[layersCount][][];
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
                _weights[i] = new double[currentLayer.Length][];
                for (int j = 0; j < _weights[i].Length; j++)
                {
                    _weights[i][j] = new double[prevLayer.Length];
                    RandomInitializeWeights(i,j);
                }
            }
            this._isBuilt = true;
        }
        private void RandomInitializeWeights(int layerIdx,int neuronIdx)
        {
            var weights = _weights[layerIdx][neuronIdx];
            for(int i = 0;i<weights.Length;i++)
                weights[i] = (Utils.Utils.Random.NextDouble() * 2) - 1;
        }
        public IReadOnlyCollection<double> Predict(double[] input)
        {
            return GetDataFromForward(input).Last().Value;
        }

        public IReadOnlyCollection<double> Predict(IEnumerable<double> input)
        {
            return this.Predict(input.ToArray());
        }

        public void Train(double[][] batch, double[][] validResult, int epochs, double learningRate = 0.1)
        {
            Console.WriteLine("Starting Train!");
            for(int e = 0; e < epochs; e++)
            {
                double totalError = 0.0;
                for(int i = 0; i < batch.Length; i++)
                {
                    var data = batch[i];
                    var validPrediction = validResult[i];
                    var networkPrediction = Predict(data);
                    totalError += MSE(networkPrediction, validPrediction);
                    _Train(data, validPrediction, learningRate);
                }
                Console.WriteLine($"Epoch {e + 1}/{epochs}: Total Error: {totalError}");
            }
        }

        public void Train(double[,] batch, double[,] validResult, int epochs, double learningRate = 0.1)
        {
            var batchConverted = ConvertMatrixToDoubleArray(batch);
            var validResConverted = ConvertMatrixToDoubleArray(validResult);
            this.Train(batchConverted,validResConverted, epochs, learningRate);   
        }
        private double[][] ConvertMatrixToDoubleArray(double[,] matrix)
        {
            var cache = new double[matrix.GetLength(0)][];
            for(int i = 0;i<matrix.GetLength(0);i++)
                for(int j = 0;j<matrix.GetLength(1);j++)
                    cache[i][j] = matrix[i,j];
            return cache;
        }

        public void Train(IEnumerable<IEnumerable<double>> batch, IEnumerable<IEnumerable<double>> validResult, int epochs, double learningRate = 0.1)
        {
            var batchConverted = batch.Select(e=>e.ToArray()).ToArray();
            var validResConverted = validResult.Select(e=>e.ToArray()).ToArray();
            this.Train(batchConverted, validResConverted, epochs, learningRate);  
        }
        private void _Train(double[] input, double[] validPrediction,double learningRate)
        {
            #region Backpropagation Algorithm
            var forwardData = GetDataFromForward(input);
            var networkPrediction = forwardData.Last().Value;
            var lastGradient = new double[1];

            _TrainOutputLayer(networkPrediction, validPrediction, ref lastGradient, learningRate);
            for (int i = _layers.Length - 2; i > 0; i--)
                _TrainHiddenLayer(i, forwardData[i], ref lastGradient,learningRate);
            _TrainInputLayer(lastGradient, forwardData[0], learningRate);
            #endregion

        }
        private void UpdateWeights(int layerIdx, double[] gradient, double[] forwardData,double learningRate)
        {
            for(int i = 0;i< _weights[layerIdx + 1].Length; i++) // layerWeights
                for (int j = 0; j < _weights[layerIdx + 1][i].Length; j++) // neuronWeights
                    _weights[layerIdx + 1][i][j] += learningRate * gradient[i] * forwardData[j];
        }
        private void _TrainInputLayer(double[] gradient, double[] forwardData,double learningRate)
        {
            //UpdateWeights(0, gradient, forwardData, learningRate);
        }
        private void _TrainHiddenLayer(int layerIdx, double[] forwardData,ref double[] lastGradient,double learningRate)
        {
            //UpdateWeights(layerIdx, lastGradient, forwardData, learningRate);
            var backwardData = BackwardThroughNetwork(lastGradient, layerIdx);
            lastGradient = CalculateGradient(backwardData, forwardData, layerIdx);
            //UpdateBiases(layerIdx, lastGradient, learningRate);
        }
        private void _TrainOutputLayer(double[] networkPrediction, double[] validPrediction,ref double[] lastGradient,double learningRate)
        {
            var outputError = CalculateOutputError(networkPrediction, validPrediction);
            //var outputError = CalculateOutputErrorSOFTMAX(networkPrediction, validPrediction);
            //lastGradient = outputError;
            lastGradient = CalculateGradient(outputError, networkPrediction, _layers.Length - 1);
            //UpdateBiases(_layers.Length - 1, lastGradient, learningRate);

        }
        private void UpdateBiases(int layerIdx, double[] gradient,double learningRate)
        {
            for(int i = 0;i < _layers[layerIdx].Length;i++)
                _layers[layerIdx][i] += gradient[i] * learningRate;
        }
        private double[] CalculateOutputError(double[] networkPrediction, double[] validPrediction)
        {
            var cache = new double[networkPrediction.Length];
            for(int i = 0;i< cache.Length;i++)
                cache[i] = validPrediction[i] - networkPrediction[i];
            return cache;
        }
        private double[] CalculateOutputErrorSOFTMAX(double[] networkPrediction, double[] validPrediction)
        {
            var cache = new double[networkPrediction.Length];
            for (int i = 0; i < cache.Length; i++)
                cache[i] = networkPrediction[i] - validPrediction[i];
            return cache;
        }
        private double[] CalculateGradient(double[] backwardInput, double[] forwardData,int layerIdx)
        {
            var cache = new double[backwardInput.Length];
            EraseActivationFromData(ref forwardData, layerIdx);
            for(int i = 0; i<cache.Length;i++)
                cache[i] = backwardInput[i] * forwardData[i];
            return cache;
        }

        private double[] BackwardThroughNetwork(double[] input, int layerIdx)
        {
            var layerWeights = _weights[layerIdx + 1];
            var layer = _layers[layerIdx];
            var result = new double[layer.Length];


            for (int n = 0; n < layer.Length; n++)
            {
                var nRez = 0.0;
                for (int i = 0; i < input.Length; i++)
                    nRez += input[i] * layerWeights[i][n]; //BUG
                result[n] = nRez;
            }

            return result;
        }


        private Dictionary<int, double[]> GetDataFromForward(double[] input)
        {
            var _tmpForwardSave = new Dictionary<int, double[]>();
            var _tmpData = input;

            _tmpForwardSave.Add(0, input);
            for(int i = 1; i < _layers.Length; i++)
                _Forward(i, ref _tmpData, ref _tmpForwardSave);

            return _tmpForwardSave;
        }
        private void _Forward(int layerIdx, ref double[] tempData, ref Dictionary<int, double[]>cache)
        {
            var layer = _layers[layerIdx];
            var weights = _weights[layerIdx];
            tempData = ForwardThroughNetwork(tempData, layer, weights);
            ApplyActivationFunction(_aFunctions[layerIdx], ref tempData);
            cache.Add(layerIdx, tempData);
        }
        private double[] ForwardThroughNetwork(double[] input, double[] layer, double[][] weights)
        {
            double[] result = new double[layer.Length];
            for(int n=0;n<layer.Length; n++)
            {
                var sum = layer[n];
                var nWeights = weights[n];

                for (int i = 0; i < input.Length; i++)
                    sum += input[i] * nWeights[i];
                result[n] = sum;
            }
            return result;
        }
        private double MSE(IEnumerable<double> prediction, IEnumerable<double> outcome)
        {
            double sum = 0.0;
            for (int i = 0; i < prediction.Count(); i++)
                sum += Math.Pow(prediction.ElementAt(i) - outcome.ElementAt(i), 2);
            return sum;
        }
    }
}
