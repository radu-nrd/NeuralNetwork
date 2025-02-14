using NN_MODEL.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_MODEL.Models
{
    public class Layer : ILayer
    {
        private readonly int _index;
        private readonly List<INeuron> _neurons;
        public int Index => _index;
        public IReadOnlyCollection<INeuron> Neurons => _neurons;

        public Layer(int index)
        {
            _index = index;
            _neurons = new List<INeuron>();
        }

        public void AppendNeurons(params INeuron[] neurons)
        {
            if(neurons is null)
                throw new ArgumentNullException("neurons was null");

            foreach(var neuron in neurons)
                if(!(neuron is null))
                    _neurons.Add(neuron);
        } 
    }
}
