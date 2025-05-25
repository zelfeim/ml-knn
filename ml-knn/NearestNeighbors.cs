using ml_knn.Metrics;

namespace ml_knn;

public class NearestNeighbors<TClassification>(
    IDataFactory<IData<TClassification>, TClassification> dataFactory
)
    where TClassification : struct, Enum
{
    private List<IData<TClassification>> DataSet { get; } = [];

    private IDataFactory<IData<TClassification>, TClassification> DataFactory { get; } =
        dataFactory;

    public void LoadCsvData(string csvPath)
    {
        using var reader = new StreamReader(csvPath);

        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line))
                continue;
            var values = line.Split(',');

            DataSet.Add(DataFactory.CreateData(values.SkipLast(1).ToList(), values.Last()));
        }
    }

    public TClassification? Classify(int k, IData<TClassification> data, IMetric metric)
    {
        return Classifier.Classify<TClassification, IData<TClassification>>(
            k,
            data,
            DataSet,
            metric
        );
    }

    public TClassification? Classify(
        int k,
        IData<TClassification> data,
        IMetric metric,
        List<IData<TClassification>> dataSet
    )
    {
        return Classifier.Classify<TClassification, IData<TClassification>>(
            k,
            data,
            dataSet,
            metric
        );
    }

    public void TestClassification(int k, IMetric metric)
    {
        var errorCount = 0;
        var misses = 0;

        foreach (var data in DataSet)
        {
            var testSet = DataSet.Except([data]).ToList();
            var classification = Classify(k, data, metric, testSet);
            var expectedClassification = data.Classification;

            if (classification == null)
            {
                misses++;
                continue;
            }

            if (!classification.Equals(expectedClassification))
            {
                errorCount++;
            }
        }

        var errorRate = errorCount / (float)DataSet.Count;
        var coverage = (DataSet.Count - misses) / (float)DataSet.Count;

        Console.WriteLine($"Error rate: {errorRate}");
        Console.WriteLine($"Coverage: {coverage}");
    }
}

public interface IDataFactory<out TData, TClassification>
    where TData : IData<TClassification>
    where TClassification : struct, Enum
{
    public TData CreateData(List<string> values, string classification);
}

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
        return typeof(TData)
            .GetProperties()
            .Where(p => p.PropertyType == typeof(double) && p.GetValue(data) != null)
            .Select(p => (double)p.GetValue(data)!)
            .ToList();
    }
}

public interface IData<TClassification>
    where TClassification : struct, Enum
{
    public TClassification Classification { get; set; }
}
