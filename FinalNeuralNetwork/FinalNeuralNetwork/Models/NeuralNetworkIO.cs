using FinalNeuralNetwork.Exceptions;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Models
{
    sealed partial class NeuralNetwork
    {
        public void Save(string filePath)
        {
            if(string.IsNullOrEmpty(filePath))
                throw new ArgumentNullException("filePath is null or empty!");
            if(!Path.GetExtension(filePath).ToLower().Trim().Equals(".nn"))
                throw new InvalidFileExtensionException("The given file is not an *.nn file!");

            var jsonBuilder = new NeuralNetworkJsonBuilder(this);
            jsonBuilder.BuildJsonFile(filePath);
            var data = File.ReadAllBytes(filePath);
            for (int i = 0; i < data.Length; i++)
                data[i] = (byte)(data[i] + 26);
            File.WriteAllBytes(filePath, data);
        }

    }
}
