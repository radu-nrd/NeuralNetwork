using NN_MODEL.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_MODEL.Models
{
    public class Neuron : INeuron
    {
        private readonly string _tag;
        private double _bias;
        private readonly ILayer _parentLayer;
        public string Tag => $"N{LayerNumber}{LayerPosition}";

        public int LayerNumber => _parentLayer.Index;

        public int LayerPosition => _parentLayer.Neurons.Select((item,index)=>new { item,index}).FirstOrDefault(x=>x.item.Equals(this)).index+1;

        public double Bias => _bias;

        public Neuron(ILayer parentLayer)
        {
            _parentLayer = parentLayer;
            _bias = (Utility.Rand.NextDouble() * 2) - 1;
        }
        public Neuron(ILayer parentLayer, double bias)
        {
            _parentLayer = parentLayer;
            _bias = bias;
        }
        
        public void AdjustBias(double adjustValue)
        {
            this._bias+=adjustValue;
        }

        public void SetBias(double bias)
        {
            this._bias = bias;
        }
    }
}
