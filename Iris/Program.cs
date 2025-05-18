// See https://aka.ms/new-console-template for more information

using Iris;
using ml_knn;

var irisDataFactory = new IrisDataFactory();
var irisClassifier = new IrisClassifier();

var knnAlgorithm = new NearestNeighbors<IrisClassification>(irisDataFactory, irisClassifier);

// Verification loop

// Program loop to pass some value to classify
