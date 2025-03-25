using FinalNeuralNetwork.Utils;
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
            }
        }
        void ApplySigmoid(ref double[] data)
        {
            for(var i = 0; i < data.Length; i++)
                data[i] = Sigmoid(data[i]);
        }
        double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    }
}
