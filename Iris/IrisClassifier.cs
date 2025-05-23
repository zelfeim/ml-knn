using ml_knn;
using ml_knn.Metrics;

namespace Iris;

public class IrisClassifier : IClassifier<IrisClassification>
{
    public IrisClassification Classify(
        int k,
        IData<IrisClassification> data,
        List<IData<IrisClassification>> dataSet,
        IMetric metric
    )
    {
        var irisData = (IrisData)data;
        var irisDataSet = dataSet.Cast<IrisData>();

        List<double> xValues = [irisData.X, irisData.Y, irisData.Z];

        var neighboursWithDistance = irisDataSet
            .Select(neighbour =>
                (neighbour, metric.Calculate(xValues, [neighbour.X, neighbour.Y, neighbour.Z]))
            )
            .ToList();
        neighboursWithDistance.Sort((x, y) => x.Item2.CompareTo(y.Item2));
        var closestNeighbours = neighboursWithDistance.Take(k);

        var classifications = closestNeighbours
            .GroupBy(s => s.neighbour.Classification)
            .Select(g => new { Status = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count)
            .ToDictionary(x => x.Status, x => x.Count);

        if (classifications.Count == 0) return default;

        var classification = classifications.First();
        if (classifications.Count(c => c.Value == classification.Value) != 1) return default;

        return classification.Key;
    }
}