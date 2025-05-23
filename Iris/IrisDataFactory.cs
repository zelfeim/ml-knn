using ml_knn;
using ml_knn.Helpers;

namespace Iris;

public class IrisDataFactory : IDataFactory<IrisData, IrisClassification>
{
    public IrisData CreateData(List<string> values, string classification)
    {
        if (values.Count != 4)
            throw new ArgumentException("Invalid data");

        var castedValues = values.Select(double.Parse).ToList();
        var min = castedValues.Min();
        var max = castedValues.Max();

        var normalizedValues = castedValues
            .Select(x => ValueNormalizer.Normalize(x, min, max))
            .ToList();

        return new IrisData
        {
            X = normalizedValues[0],
            Y = normalizedValues[1],
            Z = normalizedValues[2],
            Classification = Enum.Parse<IrisClassification>(classification)
        };
    }
}