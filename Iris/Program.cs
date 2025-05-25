// See https://aka.ms/new-console-template for more information

using Iris;
using ml_knn;
using ml_knn.Metrics;

const int k = 30;

var irisDataFactory = new IrisDataFactory();

var knnAlgorithm = new NearestNeighbors<IrisClassification>(irisDataFactory);

knnAlgorithm.LoadCsvData("iris.csv");

// Verification loop

knnAlgorithm.TestClassification(k, new EuclideanMetric());

// Program loop to pass some value to classify