namespace ml_knn.Metrics;

public class EuclideanMetric : MetricBase
{
    public override double Calculate(List<double> xValues, List<double> yValues)
    {
        Validate(xValues, yValues);

        var sum = 0.0;
        for (var i = 0; i < xValues.Count; i++)
            sum += Math.Pow(xValues[i] - yValues[i], 2);

        return Math.Sqrt(sum);
    }
}
