using FinalNeuralNetwork.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Models
{
    public class Neural_Network_GPU_Setup
    {
        public required double[] Batch {  get; set; }
        public required int[] BatchOffsets { get; set; }
        public required double[] ValidPredictions { get; set; }
        public required int[] ValidPredictionOffsets {  get; set; }
        public required double[] Layers {  get; set; }
        public required int[] LayerOffsets { get; set; }
        public required double[] Weights { get; set; }
        public required int[] ActivationFunctions {  get; set; }
        public required double[] ForwardData {  get; set; }

    }
}
