using ILGPU;
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
            ArrayView1D<double, Stride1D.Dense> outcome,
            ArrayView1D<int, Stride1D.Dense> outcomeOffsets,
            ArrayView1D<double, Stride1D.Dense> layers,
            ArrayView1D<int, Stride1D.Dense> layerOffsets,
            ArrayView1D<double, Stride1D.Dense> weights,
            ArrayView1D<int, Stride1D.Dense> activationFunctions,
            ArrayView1D<double, Stride1D.Dense> forwardData
        )
        {
            var batchStartIndex = batchOffsets[idx];
            var batchEndIndex = batchOffsets[idx + 1];

            for (int i = 1; i < layers.Length; i++)
            {
                for (int j = layerOffsets[i]; j < layerOffsets[i + 1]; j++)
                {
                    //TODO build setup , too hard to understand rn
                }
            }

        }
    }
}
