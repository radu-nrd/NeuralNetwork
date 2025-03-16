using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU;

namespace NN_MODEL
{
    public static class Kernel
    {
        public static void Forward(
            Index1D idx,
            ArrayView<float>input,
            ArrayView<float>biases,
            ArrayView<float>weights,
            ArrayView<float>result) 
        {
            float sum = biases[idx];
            for(int i=0;i<input.Length;i++)
                sum += input[i] * weights[i];
            sum = (float)(1.0f / (1.0f + Math.Exp(-sum))); // Sigmoid activation
            result[idx] = sum;
        }

    }
}
