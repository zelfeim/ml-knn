namespace ml_knn.Metrics;

public interface IMetric
{
    public double Calculate(List<double> xValues, List<double> yValues);
}