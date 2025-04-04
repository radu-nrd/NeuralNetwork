using FinalNeuralNetwork.Utils;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Models
{
    public partial class NeuralNetwork
    {
        CLAccelerator _graphicsAccelerator;

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
            ArrayView1D<double,Stride1D.Dense>
            > _trainKernel;
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
            ArrayView1D<double, Stride1D.Dense>
            > (GPU_KERNEL._Train);
        }

        private Neural_Network_GPU_Setup BuildGpuModel(double[][] batch, double[][] validResult)
        {
            return new Neural_Network_GPU_Setup
            {
                Batch = BuildFlatArrayFrom2D(batch),
                BatchOffsets = BuildOffsetForArray2D(batch),
                ValidPredictions = BuildFlatArrayFrom2D(validResult),
                ValidPredictionOffsets = BuildOffsetForArray2D(validResult),
                Layers = BuildFlatArrayFrom2D(_layers),
                LayersCount = _layers.Select(l=>l.Length).ToArray(),
                Weights = BuildFlatArrayFrom3D(_weights),
                ActivationFunctions = [],
                ForwardData = new double[_layers.Select(l => l.Length).ToArray().Sum() - _layers[0].Length],
                Input = new double[_layers.Select(l => l.Length).ToArray().Max()],
                Gradient = new double[_layers.Select(l => l.Length).ToArray().Sum() - _layers[0].Length]
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
            int[] offsets = new int[array.Length];
            offsets[0] = 0;
            for(int  i = 1; i < array.Length; i++)
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

        public void TrainGPU(double[][] batch, double[][] validResult, int epochs)
        {
            GPUSetup();
            var model = BuildGpuModel(batch, validResult);
            for(int i = 0;i<batch.Length;i++)
                _TestKernel(i,model);
            throw new NotImplementedException();
        }
        private void _TestKernel(int idx,Neural_Network_GPU_Setup setup)
        {
            #region Indexes
            var forwardDataIndex = 0;
            var batchDataStartIndex = setup.BatchOffsets[idx];
            var batchDataEndIndex = setup.BatchOffsets[idx + 1];
            var inputIndex = 0;
            var neuronStartIndex = setup.LayersCount[0];
            var weightStartIndex = 0;
            var validPredictionStartIndex = setup.ValidPredictionOffsets[idx];
            var validPredictionEndIndex = setup.ValidPredictionOffsets[idx + 1];
            var gradientIndex = setup.ForwardData.Length-1;
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
                    inputIndex = 0;

                    for(int k = weightStartIndex; k < weightEndIndex; k++)
                    {
                        setup.ForwardData[forwardDataIndex] += setup.Weights[k] * setup.Input[inputIndex];
                        inputIndex++;
                    }
                    setup.ForwardData[forwardDataIndex] = Sigmoid(setup.ForwardData[forwardDataIndex]);
                    forwardDataIndex++;
                    weightStartIndex = weightEndIndex;
                }

                for(int k = 0;k < neuronEndIndex - neuronStartIndex; k++)
                {
                    setup.Input[k] = setup.ForwardData[saveStartForwardingIndex];
                    saveStartForwardingIndex++;
                }

                neuronStartIndex = neuronEndIndex;
            }
            #endregion

            #region Backpropagation Algorithm
            #region CalculateOutputGradient
            var networkPredictionStartIndex = setup.ForwardData.Length - setup.LayersCount[setup.LayersCount.Length - 1];

            for(int i = validPredictionStartIndex; i < validPredictionEndIndex; i++)
            {
                setup.Gradient[gradientIndex] = setup.ValidPredictions[i] - setup.ForwardData[networkPredictionStartIndex + i];
                setup.Gradient[gradientIndex] *= SigmoidDerivative(setup.ForwardData[networkPredictionStartIndex + i]);
                gradientIndex--;
            }

            #endregion
            #endregion
        }

        public void TrainGPU(double[,] batch, double[,] validResult, int epochs)
        {
            throw new NotImplementedException();
        }

        public void TrainGPU(IEnumerable<IEnumerable<double>> batch, IEnumerable<IEnumerable<double>> validResult, int epochs)
        {
            throw new NotImplementedException();
        }
    }
}
