namespace ml_knn;

public interface IData<TClassification>
    where TClassification : struct, Enum
{
    public TClassification Classification { get; set; }
}