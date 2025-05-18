using ml_knn.Metrics;

namespace ml_knn;

public class NearestNeighbors<TClassification>(
    IDataFactory<IData<TClassification>, TClassification> dataFactory,
    IClassifier<TClassification> dataClassifier
)
    where TClassification : Enum
{
    private List<IData<TClassification>> DataSet { get; } = [];
    private IClassifier<TClassification> DataClassifier { get; } = dataClassifier;
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

    public TClassification Classify(int k, IData<TClassification> data, IMetric metric)
    {
        return DataClassifier.Classify(k, data, DataSet, metric);
    }
}

public interface IDataFactory<out TData, TClassification>
    where TData : IData<TClassification>
    where TClassification : Enum
{
    public TData CreateData(List<string> values, string classification);
}

public interface IClassifier<TClassification>
    where TClassification : Enum
{
    public TClassification Classify(
        int k,
        IData<TClassification> data,
        List<IData<TClassification>> dataSet,
        IMetric metric
    );
}

public interface IData<TClassification>
    where TClassification : Enum
{
    public TClassification Classification { get; set; }
}
