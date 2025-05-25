namespace ml_knn;

public interface IDataFactory<out TData, TClassification>
    where TData : IData<TClassification>
    where TClassification : struct, Enum
{
    public TData CreateData(List<string> values, string classification);
}