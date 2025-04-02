using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Interfaces
{
    /// <summary>
    /// Interface that contains training methods using GPU for a Neural Network Interface.
    /// </summary>
    public interface ITrainableGPU
    {

        /// <summary>
        /// Start the training process of the network.
        /// </summary>
        /// <param name="batch">Batch of data</param>
        /// <param name="validResult">Valid result for each data in the batch</param>
        /// <param name="epochs">Number of iterations. How many times batch will be passed through network</param>
        void TrainGPU(double[][] batch, double[][] validResult, int epochs);

        /// <summary>
        /// Start the training process of the network.
        /// </summary>
        /// <param name="batch">Batch of data</param>
        /// <param name="validResult">Valid result for each data in the batch</param>
        /// <param name="epochs">Number of iterations. How many times batch will be passed through network</param>
        void TrainGPU(double[,] batch, double[,] validResult, int epochs);

        /// <summary>
        /// Start the training process of the network.
        /// </summary>
        /// <param name="batch">Batch of data</param>
        /// <param name="validResult">Valid result for each data in the batch</param>
        /// <param name="epochs">Number of iterations. How many times batch will be passed through network</param>
        void TrainGPU(IEnumerable<IEnumerable<double>> batch, IEnumerable<IEnumerable<double>> validResult, int epochs);
    }
}
