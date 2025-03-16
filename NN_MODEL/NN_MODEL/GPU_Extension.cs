using NN_MODEL.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Frontend;
using System.Runtime.InteropServices;

namespace NN_MODEL.Models
{
    public partial class NeuralNetworkModel
    {

        public static void UpdateWeightsKernel(
            Index1D idx,
            ArrayView<double> weights,
            ArrayView<double> gradient,
            double forwardData,
            double learningRate = 0.1
            )
        {
            if (idx > weights.IntLength)
                return;
            weights[idx] += learningRate * gradient[idx] * forwardData;

        }

        public static void GradientCalculationKernel(
            Index1D idx,
            ArrayView<double> backwardData, 
            ArrayView<double> forwardData, 
            ArrayView<double> result)
        {
            if (idx > backwardData.IntLength)
                return;
            result[idx] = backwardData[idx] * (forwardData[idx] * (1 - forwardData[idx]));
        }
        public static void UpdateBiasesKernel(
            Index1D idx,
            ArrayView<double> neurons,
            ArrayView<double> gradient,
            double learningRate = 0.1)
        {
            if (idx > neurons.IntLength)
                return;
            neurons[idx] += learningRate * gradient[idx];
        }
        
        public static void SubKernel(
            Index1D idx,
            ArrayView<double> array1, 
            ArrayView<double> array2,
            ArrayView<double>result)
        {
            if (idx > array1.Length)
                return;
            result[idx] = array1[idx] - array2[idx];
        }

        public static void NeuronProcessKernel(
            Index1D idx,
            ArrayView<double> input,
            ArrayView<double>weights,
            ArrayView<double> partialResult
            )
        {
            if (idx > input.Length)
                return;
            partialResult[idx] = input[idx] * weights[idx];
        }

        public static void ForwardKernel(
            Index1D idx,
            ArrayView<double> input,
            ArrayView<double> biases,
            ArrayView<double> weights,
            ArrayView<double> result)
        {
            double sum = biases[idx];
            for (int i = 0; i < input.Length; i++)
            {
                int weightIndex = idx * input.IntLength + i; // Compute correct weight index
                sum += input[i] * weights[weightIndex];
            }
                
            sum = (float)(1.0f / (1.0f + XMath.Exp(-sum))); // Sigmoid activation
            result[idx] = sum;
        }


        public NN_GPU_MODEL GPU_MODEL { get; set; }
        public CLAccelerator Accelerator { get; set; }
        public Action<Index1D, ArrayView<double>,ArrayView<double>, ArrayView<double>> NeuronProcessKernelObj {  get; set; }
        public Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>> SubKernelObj { get; set; }
        public Action<Index1D, ArrayView<double>, ArrayView<double>, double> UpdateBiasesKernelObj {  get; set; }
        public Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>> GradientCalculationKernelObj { get; set; }
        public Action<Index1D,ArrayView<double>,ArrayView<double>,double,double> UpdateWeightsKernelObj {  get; set; }

        public void Setup()
        {
            GPU_MODEL = CastToGpuModel();
            Accelerator = Context.Create(builder => builder.OpenCL().EnableAlgorithms()).CreateCLAccelerator(0);
            NeuronProcessKernelObj = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(NeuronProcessKernel);
            SubKernelObj = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(SubKernel);
            UpdateBiasesKernelObj = Accelerator.LoadAutoGroupedStreamKernel < Index1D, ArrayView<double>, ArrayView<double>, double>(UpdateBiasesKernel);
            GradientCalculationKernelObj = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(GradientCalculationKernel);
            UpdateWeightsKernelObj = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, double,double>(UpdateWeightsKernel);
        }

        public void TrainOnGPU(double[][] batch, double[][] outcome, int epochs)
        {
            Console.WriteLine("Starting Train!");
            //var idx = 1;
            //for(int i = 0; i < GPU_MODEL.Weights.Length; i++)
            //{
            //    var weights = GPU_MODEL.Weights[i];
            //    for(int j = 0; j < weights.Length; j++)
            //    {
            //        Console.WriteLine($"W{idx} : {weights[j]}");
            //        idx++;
            //    }
            //}
            //Console.WriteLine();
            for (int e = 0; e < epochs; e++)
            {
                double totalError = 0.0;
                for (int i = 0; i < batch.Length; i++)
                {
                    var data = batch[i];
                    var valid_prediction = outcome[i];
                    var network_prediction = PredictGPU(data);
                    totalError += MSE(network_prediction, valid_prediction);
                    TrainGPU(data, valid_prediction);
                }
                Console.WriteLine($"Epoch {e + 1}/{epochs}: Total Error: {totalError}");
                //idx = 1;
                //for (int i = 0; i < GPU_MODEL.Weights.Length; i++)
                //{
                //    var weights = GPU_MODEL.Weights[i];
                //    for (int j = 0; j < weights.Length; j++)
                //    {
                //        Console.WriteLine($"W{idx} : {weights[j]}");
                //        idx++;
                //    }
                //}
                //Console.WriteLine();
            }
        }
        private void TrainGPU(IEnumerable<double> input, IEnumerable<double> outcome)
        {
            var _tmpForwardSave = GetDataFromForward_GPU(input);
            var prediction = _tmpForwardSave.Last().Value;

            var outcomeBuffer = Accelerator.Allocate1D(outcome.ToArray());
            var predictionBuffer = Accelerator.Allocate1D(prediction.ToArray());
            var resultBuffer = Accelerator.Allocate1D<double>(outcome.Count());
            SubKernelObj(outcome.Count(), outcomeBuffer.View, predictionBuffer.View, resultBuffer.View);
            Accelerator.Synchronize();

            var outputError = resultBuffer.GetAsArray1D();
            var outputErrorBuffer = Accelerator.Allocate1D(outputError.ToArray());
            resultBuffer = Accelerator.Allocate1D<double>(outputError.Length);

            GradientCalculationKernelObj(outputError.Length, outputErrorBuffer.View, predictionBuffer.View, resultBuffer.View);
            Accelerator.Synchronize();
            var lastGradient = resultBuffer.GetAsArray1D();
            var lastNeurons = GPU_MODEL.Layers.Last();

            var lastNeuronsBuffer = Accelerator.Allocate1D(lastNeurons.ToArray());
            var lastGradientBuffer = Accelerator.Allocate1D(lastGradient.ToArray());

            UpdateBiasesKernelObj(lastNeurons.Count(), lastNeuronsBuffer.View, lastGradientBuffer.View, 0.1);
            Accelerator.Synchronize();
            GPU_MODEL.Layers[GPU_MODEL.Layers.Length - 1] = lastNeuronsBuffer.GetAsArray1D();
            for (int i = GPU_MODEL.Layers.Length - 2; i > 0; i--)
            {
                var weights = GPU_MODEL.Weights[i + 1];
                var forwardData = _tmpForwardSave[i].ToArray();
                var neurons = GPU_MODEL.Layers[i];
                var inputFromBackward = new double[neurons.Length];
                lastGradientBuffer = Accelerator.Allocate1D(lastGradient);
                var newWeightsUpdated = new List<double>();

                for (int j = 0; j < neurons.Length; j++)
                {
                    var partialWeights = weights.Skip(j * lastGradient.Length).Take(lastGradient.Length).ToArray();
                    var partialWeightsBuffer = Accelerator.Allocate1D(partialWeights);

                    UpdateWeightsKernelObj(partialWeights.Length, partialWeightsBuffer.View, lastGradientBuffer.View, forwardData[j], 0.1);
                    Accelerator.Synchronize();
                    partialWeights = partialWeightsBuffer.GetAsArray1D();
                    newWeightsUpdated.AddRange(partialWeights);

                    //backpropagation
                    var partialResultbuffer = Accelerator.Allocate1D<double>(lastGradient.Length);
                    NeuronProcessKernelObj(partialWeights.Length, lastGradientBuffer.View, partialWeightsBuffer.View, partialResultbuffer.View);
                    Accelerator.Synchronize();
                    inputFromBackward[j] = partialResultbuffer.GetAsArray1D().Sum();
                }
                //Actual Update
                GPU_MODEL.Weights[i + 1] = newWeightsUpdated.ToArray();

                var inputFromBackwardBuffer = Accelerator.Allocate1D(inputFromBackward);
                var forwardDataBuffer = Accelerator.Allocate1D(forwardData);
                resultBuffer = Accelerator.Allocate1D<double>(forwardData.Length);

                GradientCalculationKernelObj(inputFromBackward.Length, inputFromBackwardBuffer.View, forwardDataBuffer.View, resultBuffer.View);
                Accelerator.Synchronize();
                lastGradient = resultBuffer.GetAsArray1D();

                var neuronsBuffer = Accelerator.Allocate1D(neurons);
                UpdateBiasesKernelObj(neurons.Length, neuronsBuffer.View, resultBuffer.View, 0.1);
                Accelerator.Synchronize();
                GPU_MODEL.Layers[i] = neuronsBuffer.GetAsArray1D();
            }

            var firstNeurons = GPU_MODEL.Layers.First();
            lastGradientBuffer = Accelerator.Allocate1D(lastGradient);
            var firstSetOfWeights = GPU_MODEL.Weights[1];
            var firstForwardData = _tmpForwardSave[0].ToArray();
            var newWeightsUpdate = new List<double>();

            for (int i = 0; i < firstNeurons.Length; i++)
            {
                var partialWeights = firstSetOfWeights.Skip(i * lastGradient.Length).Take(lastGradient.Length).ToArray();
                var partialWeightsBuffer = Accelerator.Allocate1D(partialWeights);

                UpdateWeightsKernelObj(partialWeights.Length, partialWeightsBuffer.View, lastGradientBuffer.View, firstForwardData[i], 0.1);
                Accelerator.Synchronize();

                partialWeights = partialWeightsBuffer.GetAsArray1D();
                newWeightsUpdate.AddRange(partialWeights);
            }
            GPU_MODEL.Weights[1] = newWeightsUpdate.ToArray();
        }

        public IEnumerable<double> PredictGPU(IEnumerable<double> input)
        {
            return GetDataFromForward_GPU(input).Last().Value;
        }

        public Dictionary<int, IEnumerable<double>> GetDataFromForward_GPU(IEnumerable<double> input)
        {
            var _tmpForwardSave = new Dictionary<int,IEnumerable<double>>();
            var _tmpData = input;
            _tmpForwardSave.Add(0, input);
            for(int i = 1; i < GPU_MODEL.Layers.Length; i++)
            {
                var biases = GPU_MODEL.Layers[i];
                var weights = GPU_MODEL.Weights[i];
                var saveResults = new double[biases.Length];
                
                for (int j = 0; j < biases.Length; j++)
                {
                    var bias = biases[j];
                    var partialWeights = weights.Skip(j* _tmpData.Count()).Take(_tmpData.Count()).ToArray();
                    var partialResultbuffer = Accelerator.Allocate1D<double>(_tmpData.Count());
                    var inputBuffer = Accelerator.Allocate1D(_tmpData.ToArray());
                    var weightsBuffer = Accelerator.Allocate1D(partialWeights);
                    NeuronProcessKernelObj(partialWeights.Length, inputBuffer.View,weightsBuffer.View, partialResultbuffer.View);
                    Accelerator.Synchronize();
                    saveResults[j] = partialResultbuffer.GetAsArray1D().Sum()+bias;
                    saveResults[j] = 1.0 / (1.0 + Math.Exp(-saveResults[j]));
                }
                _tmpData = saveResults;
                _tmpForwardSave.Add(i, _tmpData);
            }
            return _tmpForwardSave;

        }

        public NN_GPU_MODEL CastToGpuModel()
        {
            var gpuModel = new NN_GPU_MODEL
            {
                Layers = new double[_layers.Count][], //first array is layer, second is biases
                Weights = new double[_layers.Count][] //first is corespondent layer, second is weight values
            };

            for (int i = 0; i < gpuModel.Layers.Length; i++)
            {
                gpuModel.Layers[i] = new double[_layers[i].Neurons.Count];
                for (int j = 0; j < _layers[i].Neurons.Count; j++)
                    gpuModel.Layers[i][j] = _layers[i].Neurons.ElementAt(j).Bias;
            }
                

            for(int i = 0; i < gpuModel.Weights.Length; i++)
            {
                var weightsCache = new List<double>();
                foreach (var n in _layers[i].Neurons)
                    weightsCache.AddRange(GetWeightValues(n, Direction.Forward));
                gpuModel.Weights[i] = weightsCache.ToArray();
            }

            return gpuModel;
        }
    }
}
