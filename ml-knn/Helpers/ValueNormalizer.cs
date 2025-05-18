namespace ml_knn.Helpers;

public class ValueNormalizer
{
    public static double Normalize(double value, double min, double max)
    {
        return (value - min) / (max - min);
    }
}