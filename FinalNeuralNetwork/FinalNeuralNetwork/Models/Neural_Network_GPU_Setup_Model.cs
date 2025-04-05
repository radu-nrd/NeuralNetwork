using FinalNeuralNetwork.Interfaces;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinalNeuralNetwork.Models
{
    public class __OLD__Neural_Network_GPU_Setup
    {
        public required double[] Batch {  get; set; }
        public required int[] BatchOffsets { get; set; }
        public required double[] ValidPredictions { get; set; }
        public required int[] ValidPredictionOffsets {  get; set; }
        public required double[] Layers {  get; set; }
        public required int[] LayersCount { get; set; }
        public required double[] Weights { get; set; }
        public required int[] ActivationFunctions {  get; set; }
        public required double[] ForwardData {  get; set; }
        public required double[] Input {  get; set; } 
        public required double[] Gradient{  get; set; }

    }

    public class NeuralNetworkGpuSetup
    {
        public required MemoryBuffer1D<double,Stride1D.Dense> BatchBuffer {  get; set; }
        public required MemoryBuffer1D<int, Stride1D.Dense> BatchOffsetsBuffer { get; set; }
        public required MemoryBuffer1D<double, Stride1D.Dense> ValidPredictionsBuffer {  get; set; }
        public required MemoryBuffer1D<int, Stride1D.Dense> ValidPredictionsOffsetsBuffer {  get; set; }
        public required MemoryBuffer1D<double, Stride1D.Dense> LayersBuffer {  get; set; }
        public required MemoryBuffer1D<int, Stride1D.Dense> LayersCountBuffer { get; set; }
        public required MemoryBuffer1D<double, Stride1D.Dense> WeightsBuffer { get; set; }
        public required MemoryBuffer1D<int, Stride1D.Dense> ActivationFunctionsBuffer { get; set; }
        public required MemoryBuffer1D<double, Stride1D.Dense> ForwardDataBuffer { get; set; }
        public required MemoryBuffer1D<double, Stride1D.Dense> InputBuffer {  get; set; }
        public required MemoryBuffer1D<double, Stride1D.Dense> GradientBuffer {  get; set; }

    }
}
