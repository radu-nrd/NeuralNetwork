using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.JavaScript;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Utils
{
    internal class GPU_KERNEL
    {
        internal static void _Train
        (
            Index1D idx,
            ArrayView1D<double, Stride1D.Dense> batch,
            ArrayView1D<int, Stride1D.Dense> batchOffsets,
            ArrayView1D<double, Stride1D.Dense> validPredictions,
            ArrayView1D<int, Stride1D.Dense> validPredictionOffsets,
            ArrayView1D<double, Stride1D.Dense> layers,
            ArrayView1D<int, Stride1D.Dense> layersCount,
            ArrayView1D<double, Stride1D.Dense> weights,
            ArrayView1D<int, Stride1D.Dense> activationFunctions,
            ArrayView1D<double, Stride1D.Dense> forwardData,
            ArrayView1D<double, Stride1D.Dense> input,
            ArrayView1D<double, Stride1D.Dense> gradient,
            ArrayView2D<double,Stride2D.DenseX> gradientSave
        )
        {

            if (idx > 4)
                return;

            #region Indexes
            var forwardDataIndex = 0;
            var batchDataStartIndex = batchOffsets[idx];
            var batchDataEndIndex = batchOffsets[idx + 1];
            var inputIndex = 0;
            var neuronStartIndex = layersCount[0];
            var weightStartIndex = 0;
            var validPredictionStartIndex = validPredictionOffsets[idx];
            var validPredictionEndIndex = validPredictionOffsets[idx + 1];
            var gradientIndex = forwardData.Length - 1;
            #endregion

            #region CopyBatchDataToInput
            for (int i = batchDataStartIndex; i < batchDataEndIndex; i++)
            {
                input[inputIndex] = batch[i];
                inputIndex++;
            }
            #endregion

            #region Forward Propagation Algorithm
            for (int i = 1; i < layersCount.Length; i++) //layers
            {
                var prevLayerCount = layersCount[i - 1];
                var neuronEndIndex = neuronStartIndex + layersCount[i];
                var saveStartForwardingIndex = forwardDataIndex;
                for (int j = neuronStartIndex; j < neuronEndIndex; j++)
                {
                    var weightEndIndex = weightStartIndex + prevLayerCount;
                    forwardData[forwardDataIndex] = layers[j];
                    inputIndex = 0;

                    for (int k = weightStartIndex; k < weightEndIndex; k++)
                    {
                        forwardData[forwardDataIndex] += weights[k] * input[inputIndex];
                        inputIndex++;
                    }
                    forwardData[forwardDataIndex] = 1 / (1 + XMath.Exp(-forwardData[forwardDataIndex]));
                    forwardDataIndex++;
                    weightStartIndex = weightEndIndex;
                }

                for (int k = 0; k < neuronEndIndex - neuronStartIndex; k++)
                {
                    input[k] = forwardData[saveStartForwardingIndex];
                    saveStartForwardingIndex++;
                }

                neuronStartIndex = neuronEndIndex;
            }
            #endregion

            #region Backpropagation Algorithm

            #region Calculate Output Gradient
            var networkPredictionStartIndex = forwardData.Length - layersCount[layersCount.Length - 1];
            var outputBackwardIndex = 0;
            for (int i = validPredictionStartIndex; i < validPredictionEndIndex; i++)
            {
                gradient[gradientIndex] = validPredictions[i] - forwardData[networkPredictionStartIndex + outputBackwardIndex];
                gradient[gradientIndex] *= forwardData[networkPredictionStartIndex + outputBackwardIndex] * (1 - forwardData[networkPredictionStartIndex + outputBackwardIndex]);
                gradientIndex--;
                outputBackwardIndex++;
            }

            #endregion

            #region Calculate Hidden Gradient
            neuronStartIndex = layers.IntLength - layersCount[layersCount.Length - 1] - 1;
            weightStartIndex = weights.IntLength - 1;

            var backwardDataIndex = gradient.IntLength - 1;
            for (int i = layersCount.IntLength - 2; i > 0; i--)
            {
                var prevLayerCount = layersCount[i + 1];
                var neuronEndIndex = neuronStartIndex - layersCount[i];
                var saveStartBackwardingIndex = backwardDataIndex;

                for (int j = neuronStartIndex; j > neuronEndIndex; j--)
                {
                    backwardDataIndex = saveStartBackwardingIndex;
                    var weightEndIndex = weightStartIndex - prevLayerCount;
                    for (int k = weightStartIndex; k > weightEndIndex; k--)
                    {
                        gradient[gradientIndex] = gradient[backwardDataIndex] * weights[k]; //input
                        gradient[gradientIndex] *= forwardData[j - 2] * (1 - forwardData[j - 2]);
                        backwardDataIndex--;
                    }
                    gradientIndex--;
                    weightStartIndex = weightEndIndex;
                }
            }
            #endregion

            #endregion

            #region Save Gradient Build
            gradientSave[idx, 0] = gradient[3];
            //for (int i = 0; i < gradient.IntLength; i++)
            //    gradientSave[idx.X, i] = gradient[i];
            #endregion
        }
    }
}
