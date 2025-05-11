using FinalNeuralNetwork.Models;
using ILGPU.Algorithms.MatrixOperations;
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
        /// <param name="learningRate">Impact of data on the network</param>
        void TrainGPU(double[][] batch, double[][] validResult, int epochs, double learningRate = 0.1);

        /// <summary>
        /// Start the training process of the network.
        /// </summary>
        /// <param name="batch">Batch of data</param>
        /// <param name="validResult">Valid result for each data in the batch</param>
        /// <param name="epochs">Number of iterations. How many times batch will be passed through network</param>
        /// <param name="learningRate">Impact of data on the network</param>
        void TrainGPU(double[,] batch, double[,] validResult, int epochs, double learningRate = 0.1);

        /// <summary>
        /// Start the training process of the network.
        /// </summary>
        /// <param name="batch">Batch of data</param>
        /// <param name="validResult">Valid result for each data in the batch</param>
        /// <param name="epochs">Number of iterations. How many times batch will be passed through network</param>
        /// <param name="learningRate">Impact of data on the network</param>
        void TrainGPU(IEnumerable<IEnumerable<double>> batch, IEnumerable<IEnumerable<double>> validResult, int epochs, double learningRate = 0.1);

        /// <summary>
        /// Start the training process of the network.
        /// </summary>
        /// <param name="batch">Batch of Data</param>
        /// <param name="learningRate">Impact of data on the network</param>
        /// <returns></returns>
        double TrainStepGPU(double[][] batch, double learningRate = 0.1);
        /// <summary>
        /// Automatically detect best Accelerator that your machine is able to provide.
        /// </summary>
        /// <param name="batch">Batch of data</param>
        /// <param name="validResult">Valid result for each data in the batch</param>
        /// <returns></returns>
        NeuralNetworkGpuSetup GetOptimalGPUModel(double[][] batch, double[][] validResult);
        /// <summary>
        /// Assembly the GPU part in the neural network.
        /// </summary>
        /// <param name="batch">Batch of data</param>
        /// <param name="validResult">Valid result for each data in the batch</param>
        void InitializeGPU(double[][] batch, double[][] validResult);
    }
}
