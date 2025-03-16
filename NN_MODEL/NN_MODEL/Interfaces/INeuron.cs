using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_MODEL.Interfaces
{
    public interface INeuron
    {
        string Tag { get; }
        int LayerNumber {  get; }
        int LayerPosition {  get; }
        double Bias { get; }

        void AdjustBias(double adjustValue);
        void SetBias(double bias);
    }
}
