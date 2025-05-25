using ml_knn.Metrics;

namespace ml_knn;

public static class Classifier
{
    public static TClassification? Classify<TClassification, TData>(
        int k,
        TData dataCase,
        List<IData<TClassification>> dataSet,
        IMetric metric
    )
        where TClassification : struct, Enum
        where TData : IData<TClassification>
    {
        var neighboursWithDistance = dataSet
            .Select(neighbour =>
                (
                    neighbour,
                    metric.Calculate(GetTDataParameters(dataCase), GetTDataParameters(neighbour))
                )
            )
            .ToList();
        neighboursWithDistance.Sort((x, y) => x.Item2.CompareTo(y.Item2));

        var kClosestNeighbours = neighboursWithDistance.Take(k);

        var classifications = kClosestNeighbours
            .GroupBy(s => s.neighbour.Classification)
            .Select(g => new { Status = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count)
            .ToDictionary(x => x.Status, x => x.Count);

        if (classifications.Count == 0)
            return null;

        var classification = classifications.First();
        if (classifications.Count(c => c.Value == classification.Value) != 1)
            return null;

        return classification.Key;
    }

    private static List<double> GetTDataParameters<TData>(TData data)
    {
        return data!.GetType()
            .GetProperties()
            .Where(p => p.PropertyType == typeof(double) && p.GetValue(data) != null)
            .Select(p => (double)p.GetValue(data)!)
            .ToList();
    }
}