namespace ml_knn.Metrics;

public abstract class MetricBase : IMetric
{
    public abstract double Calculate(List<double> xValues, List<double> yValues);

    protected static void Validate(List<double> xValues, List<double> yValues)
    {
        if (xValues.Count != yValues.Count)
            throw new ArgumentException("xValues and yValues must be the same length");
    }
}
