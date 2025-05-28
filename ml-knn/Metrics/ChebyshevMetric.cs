namespace ml_knn.Metrics;

public class ChebyshevMetric : MetricBase
{
    public override double Calculate(List<double> xValues, List<double> yValues)
    {
        Validate(xValues, yValues);

        var values = xValues.Select((t, i) => t - yValues[i]).ToList();

        return values.Max();
    }
}