using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Interfaces
{
    /// <summary>
    /// Interface that contains Predict methods for a Neural Network Interface
    /// </summary>
    public interface IPredictable
    {
        /// <summary>
        /// Push data through network.
        /// </summary>
        /// <param name="input">Data to be passed through network.</param>
        /// <returns>Prediction after computational operations.</returns>
        IReadOnlyCollection<double> Predict(double[] input);

        /// <summary>
        /// Push data through network.
        /// </summary>
        /// <param name="input">Data to be passed through network.</param>
        /// <returns>Prediction after computational operations.</returns>
        IReadOnlyCollection<double> Predict(IEnumerable<double> input);
    }
}
