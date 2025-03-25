using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Exceptions
{
    public class InvalidFileExtensionException : NeuralNetworkException
    {
        public InvalidFileExtensionException() : base() { }
        public InvalidFileExtensionException(string message) : base(message) { }
    }
}
