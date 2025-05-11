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
    sealed partial class NeuralNetwork
    {
        private NeuralNetworkGpuSetup gpuSetup;
        public NeuralNetworkGpuSetup GpuSetup => gpuSetup;
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
                TotalError = new double[batch.Length],
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
            for (int i = 0; i < array.Length; i++)
            {
                for (int j = 0; j < array[i].Length; j++)
                    result.Add(array[i][j]);
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
            var totalError = 0.0;
            InitializeGPU(batch, validResult);
            for (int e = 0; e < epochs; e++)
            {
                totalError = TrainStepGPU(batch,learningRate);
                Console.WriteLine($"Epoch: {e + 1}/{epochs}: {totalError}");
            }

            #region CPU TEST WITH GPU CODE
            //var modelCPU = _Old_BuildGpuModel(batch, validResult, runThreads);
            //for (int e = 0; e < epochs; e++)
            //{
            //    for (int i = 0; i < batch.Length; i++)
            //        _TestKernel(i, modelCPU);

            //    var totalErrorTest = modelCPU.TotalError.Sum();
            //    Console.WriteLine($"Epoch: {e}/{epochs} : Error: {totalErrorTest}");
            //    var validNeuronsCount = modelCPU.LayersCount.Sum() - modelCPU.LayersCount[0];

            //    for (int i = 0; i < runThreads; i++)
            //    {
            //        double[] gradients = modelCPU.Gradient.Skip(i * validNeuronsCount).Take(validNeuronsCount).ToArray();
            //        double[] forwardData = modelCPU.ForwardData.Skip(i * validNeuronsCount).Take(validNeuronsCount).ToArray();
            //        UpdateNetwork(batch[i], gradients, forwardData, learningRate);
            //    }

            //    modelCPU.Layers = BuildFlatArrayFrom2D(_layers);
            //    modelCPU.Weights = BuildFlatArrayFrom3D(_weights);
            //}
            #endregion
        }

        public double TrainStepGPU(
            double[][] batch,
            double learningRate = 0.1)
        {
            var totalError = 0.0;

            gpuSetup.Kernel
               (
                   batch.Length,
                   gpuSetup.BatchBuffer.View,
                   gpuSetup.BatchOffsetsBuffer.View,
                   gpuSetup.ValidPredictionsBuffer.View,
                   gpuSetup.ValidPredictionsOffsetsBuffer.View,
                   gpuSetup.LayersBuffer.View,
                   gpuSetup.LayersCountBuffer.View,
                   gpuSetup.WeightsBuffer.View,
                   gpuSetup.ActivationFunctionsBuffer.View,
                   gpuSetup.ForwardDataBuffer.View,
                   gpuSetup.InputBuffer.View,
                   gpuSetup.GradientBuffer.View,
                   gpuSetup.TotalErrorBuffer.View,
                   batch.Length
               );
            gpuSetup.Accelerator.Synchronize();

            totalError = gpuSetup.TotalErrorBuffer.GetAsArray1D().Sum();
            var validNeuronsCount = gpuSetup.LayersCountBuffer.GetAsArray1D().Sum() - gpuSetup.LayersCountBuffer.GetAsArray1D()[0];
            var gradients = Build2DArrayFromFlat(gpuSetup.GradientBuffer.GetAsArray1D(),validNeuronsCount);
            var forwardData = Build2DArrayFromFlat(gpuSetup.ForwardDataBuffer.GetAsArray1D(), validNeuronsCount);

            for (int i = 0; i < batch.Length; i++)
                UpdateNetwork(batch[i], gradients[i], forwardData[i], learningRate);

            double[] flatLayers = BuildFlatArrayFrom2D(_layers);
            double[] flatWeights = BuildFlatArrayFrom3D(_weights);

            gpuSetup.LayersBuffer.View.CopyFromCPU(flatLayers);
            gpuSetup.WeightsBuffer.View.CopyFromCPU(flatWeights);

            return totalError;
        }
        private T[][] Build2DArrayFromFlat<T>(T[] array, int eachArrayCount)
        {
            if (array.Length % eachArrayCount != 0)
                throw new InvalidOperationException("The given array cannot be divided by eachArrayCount!");

            T[][] cache = new T[array.Length/eachArrayCount][];
            T[] tmpArray = new T[eachArrayCount];
            tmpArray[0] = array[0];
            int matrixIdx = 0;
            for(int i = 1; i < array.Length; i++)
            {
                if (i % eachArrayCount == 0)
                {
                    cache[matrixIdx] = tmpArray;
                    tmpArray = new T[eachArrayCount];
                    matrixIdx++;
                }
                tmpArray[i % eachArrayCount] = array[i];
            }
            cache[matrixIdx] = tmpArray;
            return cache;
               
        }

        private void UpdateNetwork(double[] input,double[] gradient, double[] forwardData,double learningRate)
        {
            var networkIdx = gradient.Length - _layers[_layers.Length - 1].Length;
            #region OutputLayers
            for (int i = 0; i < _layers[_layers.Length - 1].Length; i++)
                _layers[_layers.Length - 1][i] -= learningRate * gradient[networkIdx + i];
            #endregion

            #region Hidden Layers

            for (int i = _layers.Length - 2; i > 0; i--)
            {
                var forwardDataIdx = networkIdx - _layers[i].Length;
                for (int j = 0; j < _weights[i+1].Length; j++)
                {
                    for(int k = 0; k < _weights[i + 1][j].Length; k++)
                    {
                        _weights[i + 1][j][k] -= learningRate * gradient[networkIdx + j] * forwardData[forwardDataIdx + k];
                    }
                }

                networkIdx -= _layers[i].Length;
                for (int neuronIdx = 0; neuronIdx < _layers[i].Length; neuronIdx++)
                    _layers[i][neuronIdx] -= learningRate * gradient[networkIdx + neuronIdx];

            }
            #endregion

            #region Input Layer

            for (int j = 0; j < _weights[1].Length; j++)
            {
                for (int k = 0; k < _weights[1][j].Length; k++)
                {
                    _weights[1][j][k] -= learningRate * gradient[networkIdx + j] * input[k];
                }
            }

            #endregion

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
                setup.Gradient[gradientIndex] = setup.ForwardData[networkPredictionStartIndex + outputBackwardIndex] - setup.ValidPredictions[i];
                setup.TotalError[idx] = setup.Gradient[gradientIndex] * setup.Gradient[gradientIndex];
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
                        setup.Gradient[gradientIndex] += setup.Gradient[backwardDataIndex] * setup.Weights[weightAccessorIndex]; //input
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

        public NeuralNetworkGpuSetup GetOptimalGPUModel(double[][] batch, double[][] validResult)
        {
            var accelerator = GetOptimalAccelerator();
            var numberOfThreads = batch.Length;
            return new NeuralNetworkGpuSetup
            {
                Accelerator = accelerator,
                Kernel = accelerator
                    .LoadAutoGroupedStreamKernel<
                        Index1D,
                        ArrayView1D<double, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>,
                        ArrayView1D<double, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>,
                        ArrayView1D<double, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>,
                        ArrayView1D<double, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>,
                        ArrayView1D<double, Stride1D.Dense>,
                        ArrayView1D<double, Stride1D.Dense>,
                        ArrayView1D<double, Stride1D.Dense>,
                        ArrayView1D<double, Stride1D.Dense>,
                        int
                    >(GPU_KERNEL._Train),
                BatchBuffer = accelerator.Allocate1D(BuildFlatArrayFrom2D(batch)),
                BatchOffsetsBuffer = accelerator.Allocate1D(BuildOffsetForArray2D(batch)),
                ValidPredictionsBuffer = accelerator.Allocate1D(BuildFlatArrayFrom2D(validResult)),
                ValidPredictionsOffsetsBuffer = accelerator.Allocate1D(BuildOffsetForArray2D(validResult)),
                LayersBuffer = accelerator.Allocate1D(BuildFlatArrayFrom2D(_layers)),
                LayersCountBuffer = accelerator.Allocate1D(_layers.Select(l => l.Length).ToArray()),
                WeightsBuffer = accelerator.Allocate1D(BuildFlatArrayFrom3D(_weights)),
                ActivationFunctionsBuffer = accelerator.Allocate1D([0]),
                ForwardDataBuffer = accelerator.Allocate1D(new double[numberOfThreads * (_layers.Select(l => l.Length).ToArray().Sum() - _layers[0].Length)]),
                InputBuffer = accelerator.Allocate1D(new double[numberOfThreads * _layers.Select(l => l.Length).ToArray().Max()]),
                GradientBuffer = accelerator.Allocate1D(new double[numberOfThreads * (_layers.Select(l => l.Length).ToArray().Sum() - _layers[0].Length)]),
                TotalErrorBuffer = accelerator.Allocate1D(new double[batch.Length]),
                NumberOfThreads = numberOfThreads
            };
        }

        private Accelerator GetOptimalAccelerator()
        {
            var context = Context.CreateDefault();

            var cudaAccelerator = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
            if (cudaAccelerator is not null)
                return cudaAccelerator.CreateAccelerator(context);

            var openCLAccelerator = context.Devices.FirstOrDefault(d=>d.AcceleratorType == AcceleratorType.OpenCL);
            if(openCLAccelerator is not null)
                return openCLAccelerator.CreateAccelerator(context);

            var cpuAccelerator = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.CPU);
            if(cpuAccelerator is not null)
                return cpuAccelerator.CreateAccelerator(context);

            throw new InvalidOperationException("Your machine does not have posibility to create an Accelerator");
        }

        public void InitializeGPU(double[][] batch, double[][] validResult)
        {
            gpuSetup = GetOptimalGPUModel(batch, validResult);
        }
    }
}
