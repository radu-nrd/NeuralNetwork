using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Exceptions
{
    public class NotReadyForBuildException : NeuralNetworkException
    {
        public NotReadyForBuildException()  : base() { }
        public NotReadyForBuildException(string message) : base($"Failed to build network. {message}") { }
    }
}
