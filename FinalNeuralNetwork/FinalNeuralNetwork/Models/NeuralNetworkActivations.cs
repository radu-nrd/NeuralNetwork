using FinalNeuralNetwork.Utils;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Models
{
    sealed partial class NeuralNetwork
    {
        void ApplyActivationFunction(ActivationFunction actFct, ref double[] data)
        {
            switch(actFct)
            {
                case ActivationFunction.None:
                    break;
                case ActivationFunction.Sigmoid:
                    ApplySigmoid(ref data);
                    break;
                case ActivationFunction.Softmax:
                    ApplySoftmax(ref data);
                    break;
                case ActivationFunction.Relu:
                    ApplyRelu(ref data);
                    break;
            }
        }

        private void EraseActivationFromData(ref double[] data, int layerIdx)
        {
            switch (_aFunctions[layerIdx])
            {
                case ActivationFunction.None:
                    break;
                case ActivationFunction.Sigmoid:
                    ApplySigmoidDerivative(ref data);
                    break;
                case ActivationFunction.Relu:
                    ApplyReluDerivative(ref data);
                    break;
            }
        }

        void ApplySoftmax(ref double[] data)
        {
            double max = data.Max();
            double[] exp = data.Select(x => Math.Exp(x - max)).ToArray();
            data = exp.Select(x => x / exp.Sum()).ToArray();
            //int maxIndex = Array.IndexOf(data, data.Max());

            //data = new double[data.Length];
            //data[maxIndex] = 1.0;
        }
        void ApplySoftmaxDerivative(ref double[] data,double[] oneHotTarget)
        {
            for(int i = 0; i < data.Length; i++)
                data[i] -= oneHotTarget[i];
        }
        void ApplySigmoid(ref double[] data)
        {
            for(var i = 0; i < data.Length; i++)
                data[i] = Sigmoid(data[i]);
        }
        void ApplyReluDerivative(ref double[] data)
        {
            for(int i = 0; i<data.Length;i++)
                data[i] = ReluDerivative(data[i]);
        }
        void ApplyRelu(ref double[] data)
        {
            for(int i = 0; i < data.Length; i++)
                data[i] = Relu(data[i]);
        }
        double Relu(double x) => Math.Max(0, x);
        double ReluDerivative(double x) => x>0?1:0;
        double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        double SigmoidDerivative(double x) => x * (1 - x);
        void ApplySigmoidDerivative(ref double[] data)
        {
            for(int i = 0; i < data.Length; i++)
                data[i] = SigmoidDerivative(data[i]);
        }
    }
}
