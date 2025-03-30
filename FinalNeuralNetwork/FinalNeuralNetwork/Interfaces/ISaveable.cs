using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Interfaces
{
    /// <summary>
    /// Interface that contains Save method for an object.
    /// </summary>
    public interface ISaveable
    {
        /// <summary>
        /// Write on disk data from an object.
        /// </summary>
        /// <param name="filePath">Path to the favorite destination to save file.</param>
        void Save(string filePath);
    }
}
