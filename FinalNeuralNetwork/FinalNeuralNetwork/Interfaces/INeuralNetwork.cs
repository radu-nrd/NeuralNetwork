using FinalNeuralNetwork.Exceptions;
using FinalNeuralNetwork.Models;
using FinalNeuralNetwork.Utils;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Interfaces
{
    /// <summary>
    /// Base Interface of a Neural Network Model
    /// </summary>
    public interface INeuralNetwork : ITrainable,ITrainableGPU,IPredictable,ISaveable
    {

        /// <summary>
        /// Error of network for specific data after train.
        /// </summary>
        double Error { get; }

        /// <summary>
        /// True if neural network is connected, False otherwise.
        /// </summary>
        bool IsBuilt {  get; }

        /// <summary>
        /// Append layer to the network. Biases of neurons will be created randomly
        /// </summary>
        /// <param name="neuronsCount">Number of neurons that layer should contain</param>
        void AppendLayer(int neuronsCount);

        /// <summary>
        /// Append layer to the network. Biases of neurons will be created randomly
        /// </summary>
        /// <param name="neuronsCount">Number of neurons that layer should contain</param>
        /// <param name="actv">Specific activation function</param>
        void AppendLayer(int neuronsCount, ActivationFunction actv);

        /// <summary>
        /// Append layer to the network. On Forward, it will be passed after a SIGMOID activation.
        /// </summary>
        /// <param name="layer">Collection of neurons</param>
        void AppendLayer(double[] layer);

        /// <summary>
        /// Append layer to the network. On Forward, it will be passed after a SIGMOID activation.
        /// </summary>
        /// <param name="layer">Collection of neurons</param>
        void AppendLayer(IEnumerable<double> layer);

        /// <summary>
        /// Append layer to the network.
        /// </summary>
        /// <param name="layer">Collection of neurons</param>
        /// <param name="actv">Specific activation function</param>
        void AppendLayer(double[] layer, ActivationFunction actv);

        /// <summary>
        /// Append layer to the network.
        /// </summary>
        /// <param name="layer">Collection of neurons</param>
        /// <param name="actv">Specific activation function</param>
        void AppendLayer(IEnumerable<double> layer, ActivationFunction actv);

        /// <summary>
        /// This method will connect the layers and will create weighs between neurons.
        /// </summary>
        void Build();

        /// <summary>
        /// Parse an existing network.
        /// </summary>
        /// <param name="filePath">Path to the file. Extension must be *.nn</param>
        /// <returns>Neural Network Model based on file.</returns>
        static INeuralNetwork Load(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                throw new ArgumentNullException("filePath is null or empty!");

            if (!Path.GetExtension(filePath).ToLower().Trim().Equals(".nn"))
                throw new InvalidFileExtensionException("The given file is not an *.nn file!");

            return NeuralNetwork.Read(filePath);
        }

        /// <summary>
        /// Base constructor for a model. Get the Neural Network Model Object.
        /// </summary>
        /// <param name="layersCount">Numbers of layers that network allocates</param>
        /// <returns>Model Object</returns>
        static INeuralNetwork CreateNetwork(int layersCount)
        {
            return new NeuralNetwork(layersCount);
        }
    }
}
