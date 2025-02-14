using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_MODEL.Interfaces
{
    public interface ILayer
    {
        int Index {  get; }
        IReadOnlyCollection<INeuron> Neurons { get; }
        void AppendNeurons(params INeuron[] neurons);
    }
}
