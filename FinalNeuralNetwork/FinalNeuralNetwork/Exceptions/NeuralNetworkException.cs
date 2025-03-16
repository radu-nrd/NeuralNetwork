using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Exceptions
{
    public class NeuralNetworkException : Exception
    {
        public NeuralNetworkException() : base() { }
        public NeuralNetworkException(string message) : base(message) { }
    }
}
