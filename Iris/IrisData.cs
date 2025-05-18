using ml_knn;

namespace Iris;

public class IrisData : IData<IrisClassification>
{
    public IrisClassification Classification { get; set; }

    public double X { get; set; }
    public double Y { get; set; }
    public double Z { get; set; }
}