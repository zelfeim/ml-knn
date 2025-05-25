using ml_knn;

namespace Iris;

public class IrisData : IData<IrisClassification>
{
    public double X { get; set; }
    public double Y { get; set; }
    public double Z { get; set; }
    public IrisClassification Classification { get; set; }
}
