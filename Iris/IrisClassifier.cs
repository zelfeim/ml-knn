using ml_knn;
using ml_knn.Metrics;

namespace Iris;

public class IrisClassifier : IClassifier<IrisClassification>
{
    public IrisClassification Classify(int k, IData<IrisClassification> data, List<IData<IrisClassification>> dataSet, IMetric metric)
    {
        var irisData = (IrisData) data;
        var irisDataSet = dataSet.Cast<IrisData>();
        
        List<double> xValues = [irisData.X, irisData.Y, irisData.Z];
        
        // Calculate and sort all distances
        var distances = irisDataSet.Select(yData => metric.Calculate(xValues, [yData.X, yData.Y, yData.Z])).ToList();
        distances.Sort();

        // Take k-closes distances
        var closestNeighbours = distances.Take(k);

        // [A, B, B, C]
        // Classify the data
        var countsByStatus = closestNeighbours.GroupBy(s => s)
            .Select(g => new { Status = g.Key, Count = g.Count() })
            .ToDictionary(x => x.Status, x => x.Count);
    }
}