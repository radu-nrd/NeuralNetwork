using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Exceptions
{
    public class InvalidAppendLayerException : NeuralNetworkException
    {
        public InvalidAppendLayerException() : base() { }
        public InvalidAppendLayerException(string message) : base(message) { }
    }
}
