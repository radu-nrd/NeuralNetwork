using FinalNeuralNetwork.Utils;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Models
{
    public partial class NeuralNetwork
    {
        CLAccelerator? _graphicsAccelerator;

        private Action
            <
            Index1D,
            ArrayView1D<double,Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<double,Stride1D.Dense>,
            ArrayView1D<int,Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<double,Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            int
            >? _trainKernel;
        private void GPUSetup()
        {
            _graphicsAccelerator = Context.Create(b => b.OpenCL().EnableAlgorithms()).CreateCLAccelerator(0);
            _trainKernel = _graphicsAccelerator
                .LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D< double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            int
            > (GPU_KERNEL._Train);
        }

        private __OLD__Neural_Network_GPU_Setup _Old_BuildGpuModel(double[][] batch, double[][] validResult,int numberOfThreads)
        {
            return new __OLD__Neural_Network_GPU_Setup
            {
                Batch = BuildFlatArrayFrom2D(batch),
                BatchOffsets = BuildOffsetForArray2D(batch),
                ValidPredictions = BuildFlatArrayFrom2D(validResult),
                ValidPredictionOffsets = BuildOffsetForArray2D(validResult),
                Layers = BuildFlatArrayFrom2D(_layers),
                LayersCount = _layers.Select(l => l.Length).ToArray(),
                Weights = BuildFlatArrayFrom3D(_weights),
                ActivationFunctions = [],
                ForwardData = new double[numberOfThreads * (_layers.Select(l => l.Length).ToArray().Sum() - _layers[0].Length)],
                Input = new double[numberOfThreads * _layers.Select(l => l.Length).ToArray().Max()],
                Gradient = new double[numberOfThreads * (_layers.Select(l => l.Length).ToArray().Sum() - _layers[0].Length)],
                NumberOfThreads = numberOfThreads
            };
        }
        private NeuralNetworkGpuSetup BuildGpuModel(double[][] batch, double[][] validResult,int numberOfThreads)
        {
            GPUSetup();
            return new NeuralNetworkGpuSetup
            {
                BatchBuffer = _graphicsAccelerator!.Allocate1D(BuildFlatArrayFrom2D(batch)),
                BatchOffsetsBuffer = _graphicsAccelerator!.Allocate1D(BuildOffsetForArray2D(batch)),
                ValidPredictionsBuffer = _graphicsAccelerator!.Allocate1D(BuildFlatArrayFrom2D(validResult)),
                ValidPredictionsOffsetsBuffer = _graphicsAccelerator!.Allocate1D(BuildOffsetForArray2D(validResult)),
                LayersBuffer = _graphicsAccelerator!.Allocate1D(BuildFlatArrayFrom2D(_layers)),
                LayersCountBuffer = _graphicsAccelerator!.Allocate1D(_layers.Select(l => l.Length).ToArray()),
                WeightsBuffer = _graphicsAccelerator!.Allocate1D(BuildFlatArrayFrom3D(_weights)),
                ActivationFunctionsBuffer = _graphicsAccelerator!.Allocate1D([0]),
                ForwardDataBuffer = _graphicsAccelerator!.Allocate1D(new double[numberOfThreads * (_layers.Select(l => l.Length).ToArray().Sum() - _layers[0].Length)]),
                InputBuffer = _graphicsAccelerator!.Allocate1D(new double[numberOfThreads * _layers.Select(l => l.Length).ToArray().Max()]),
                GradientBuffer = _graphicsAccelerator!.Allocate1D(new double[numberOfThreads * (_layers.Select(l => l.Length).ToArray().Sum() - _layers[0].Length)]),
                NumberOfThreads = numberOfThreads
            };
        }

        private T[] BuildFlatArrayFrom3D<T>(T[][][] array)
        {
            List<T> result = new List<T>();
            for (int i = 1; i < array.Length; i++) //weights all
                for (int j = 0; j < array[i].Length; j++) //layer weights
                    for (int k = 0; k < array[i][j].Length; k++)
                        result.Add(array[i][j][k]);
            return result.ToArray();
        }

        private T[] BuildFlatArrayFrom2D<T>(T[][] array)
        {
            List<T> result = new List<T>();
            int[] offsets = new int[array.Length];

            for (int i = 0; i < array.Length; i++)
            {
                for (int j = 0; j < array[i].Length; j++)
                    result.Add(array[i][j]);
                if (i < 1)
                    offsets[i] = 0;
                else
                    offsets[i] = offsets[i - 1] + array[i-1].Length;

            }
            return result.ToArray();
        }
        private int[] BuildOffsetForArray2D<T>(T[][] array)
        {
            int[] offsets = new int[array.Length+1];
            offsets[0] = 0;
            for(int  i = 1; i <= array.Length; i++)
                offsets[i] = offsets[i - 1] + array[i - 1].Length;
            return offsets;
        }

        private double[,] TranslateToMatrix(double[][] data)
        {
            var rez = new double[data.Length, data[0].Length];
            for (int i = 0; i < data.Length; i++)
                for (int j = 0; j < data[i].Length; j++)
                    rez[i, j] = data[i][j];
            return rez;
        }

        public void TrainGPU(double[][] batch, double[][] validResult, int epochs, double learningRate = 0.1)
        {
            var runThreads = batch.Length;

            var model = BuildGpuModel(batch, validResult, runThreads);
            var watch = Stopwatch.StartNew();
            for (int e = 0; e < epochs; e++)
            {
                Console.WriteLine($"Epoch: {e + 1}/{epochs}");
                _trainKernel!
                (
                    runThreads,
                    model.BatchBuffer.View,
                    model.BatchOffsetsBuffer.View,
                    model.ValidPredictionsBuffer.View,
                    model.ValidPredictionsOffsetsBuffer.View,
                    model.LayersBuffer.View,
                    model.LayersCountBuffer.View,
                    model.WeightsBuffer.View,
                    model.ActivationFunctionsBuffer.View,
                    model.ForwardDataBuffer.View,
                    model.InputBuffer.View,
                    model.GradientBuffer.View,
                    runThreads
                );
                _graphicsAccelerator!.Synchronize();
                var gradient = model.GradientBuffer.GetAsArray1D();
            }
            Console.WriteLine($"Done! Time Elapsed: {watch.ElapsedMilliseconds} ms");
            watch.Stop();

            //var modelCPU = _Old_BuildGpuModel(batch, validResult, runThreads);
            //for (int i = 0; i < batch.Length; i++)
            //    _TestKernel(i, modelCPU);
        }
        private void _TestKernel(int idx,__OLD__Neural_Network_GPU_Setup setup)
        {
            #region Indexes
            var forwardDataIndex = idx * (setup.ForwardData.Length/setup.NumberOfThreads);
            var batchDataStartIndex = setup.BatchOffsets[idx];
            var batchDataEndIndex = idx == setup.NumberOfThreads-1? setup.Batch.Length : setup.BatchOffsets[idx + 1];
            var inputIndex = idx * (setup.Input.Length / setup.NumberOfThreads);
            var neuronStartIndex = setup.LayersCount[0];
            var weightStartIndex = 0;
            var validPredictionStartIndex = setup.ValidPredictionOffsets[idx];
            var validPredictionEndIndex = idx == setup.NumberOfThreads-1? setup.ValidPredictions.Length : setup.ValidPredictionOffsets[idx + 1];
            var gradientLenght = setup.Gradient.Length / setup.NumberOfThreads;
            var forwardLenght = setup.ForwardData.Length / setup.NumberOfThreads;
            var gradientIndex = (idx * gradientLenght) + gradientLenght - 1; //starts at the end  
            #endregion

            #region CopyBatchDataToInput
            for (int i = batchDataStartIndex; i < batchDataEndIndex; i++)
            {
                setup.Input[inputIndex] = setup.Batch[i];
                inputIndex++;
            }
            #endregion

            #region Forward Propagation Algorithm
            for (int i = 1; i < setup.LayersCount.Length; i++) //layers
            {
                var prevLayerCount = setup.LayersCount[i - 1];
                var neuronEndIndex = neuronStartIndex + setup.LayersCount[i];
                var saveStartForwardingIndex = forwardDataIndex;
                for(int j = neuronStartIndex; j < neuronEndIndex; j++)
                {
                    var weightEndIndex = weightStartIndex + prevLayerCount;
                    setup.ForwardData[forwardDataIndex] = setup.Layers[j];
                    inputIndex = idx * (setup.Input.Length / setup.NumberOfThreads);

                    for(int k = weightStartIndex; k < weightEndIndex; k++)
                    {
                        setup.ForwardData[forwardDataIndex] += setup.Weights[k] * setup.Input[inputIndex];
                        inputIndex++;
                    }
                    setup.ForwardData[forwardDataIndex] = Sigmoid(setup.ForwardData[forwardDataIndex]);
                    forwardDataIndex++;
                    weightStartIndex = weightEndIndex;
                }
                inputIndex = idx * (setup.Input.Length / setup.NumberOfThreads);
                for (int k = 0;k < neuronEndIndex - neuronStartIndex; k++)
                {
                    setup.Input[inputIndex] = setup.ForwardData[saveStartForwardingIndex];
                    saveStartForwardingIndex++;
                    inputIndex++;
                }

                neuronStartIndex = neuronEndIndex;
            }
            #endregion

            #region Backpropagation Algorithm

            #region Calculate Output Gradient
            var networkPredictionStartIndex = idx * forwardLenght + (forwardLenght - setup.LayersCount[setup.LayersCount.Length - 1]);
            var outputBackwardIndex = 0;

            gradientIndex -= setup.LayersCount[setup.LayersCount.Length - 1] - 1;
            for(int i = validPredictionStartIndex; i < validPredictionEndIndex; i++)
            {
                setup.Gradient[gradientIndex] = setup.ValidPredictions[i] - setup.ForwardData[networkPredictionStartIndex + outputBackwardIndex];
                setup.Gradient[gradientIndex] *= SigmoidDerivative(setup.ForwardData[networkPredictionStartIndex + outputBackwardIndex]);
                gradientIndex++;
                outputBackwardIndex++;
            }
            gradientIndex -= setup.LayersCount[setup.LayersCount.Length - 1] + 1;
            #endregion

            #region Calculate Hidden Gradient
            neuronStartIndex = setup.Layers.Length - setup.LayersCount[setup.LayersCount.Length - 1] - 1;
            weightStartIndex = setup.Weights.Length - 1;

            var backwardDataIndex = (idx * gradientLenght) + gradientLenght - 1;
            for(int i = setup.LayersCount.Length - 2; i > 0; i--)
            {
                var neuronEndIndex = neuronStartIndex - setup.LayersCount[i];
                var saveStartBackwardingIndex = backwardDataIndex;
                var tmpSaveWeightStart = weightStartIndex;

                for (int j = neuronStartIndex; j > neuronEndIndex; j--)
                {
                    backwardDataIndex = saveStartBackwardingIndex;
                    var weightAccessorIndex = tmpSaveWeightStart;
                    for(int k = 0;k< setup.LayersCount[i+1]; k++)
                    {
                        setup.Gradient[gradientIndex] += setup.Gradient[backwardDataIndex] * setup.Weights[weightAccessorIndex]; //input //BUBAAA
                        backwardDataIndex--;
                        weightAccessorIndex -= setup.LayersCount[i];
                    }
                    setup.Gradient[gradientIndex] *= SigmoidDerivative(setup.ForwardData[(idx * forwardLenght) + j - 2]);
                    gradientIndex--;
                    tmpSaveWeightStart--;
                }
                weightStartIndex -= setup.LayersCount[i+1] * setup.LayersCount[i];
                neuronStartIndex -= setup.LayersCount[i];
            }
            #endregion

            #endregion
        }

        public void TrainGPU(double[,] batch, double[,] validResult, int epochs, double learningRate = 0.1)
        {
            throw new NotImplementedException();
        }

        public void TrainGPU(IEnumerable<IEnumerable<double>> batch, IEnumerable<IEnumerable<double>> validResult, int epochs, double learningRate = 0.1)
        {
            throw new NotImplementedException();
        }
    }
}
