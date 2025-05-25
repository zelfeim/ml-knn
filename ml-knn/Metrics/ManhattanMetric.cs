namespace ml_knn.Metrics;

public class ManhattanMetric : MetricBase
{
    public override double Calculate(List<double> xValues, List<double> yValues)
    {
        Validate(xValues, yValues);

        var sum = xValues.Select((t, i) => Math.Abs(t - yValues[i])).Sum();
        return sum;
    }
}