using System.Collections.Generic;
using ml_knn;
using ml_knn.Metrics;
using Moq;
using Xunit;

namespace Iris.Tests;

public class IrisClassifierTest
{
    [Fact]
    public void Classify_ShouldReturnCorrectClassification_WhenSingleNeighbor()
    {
        // Arrange
        var classifier = new IrisClassifier();
        var mockMetric = new Mock<IMetric>();
        mockMetric
            .Setup(m => m.Calculate(It.IsAny<List<double>>(), It.IsAny<List<double>>()))
            .Returns(0.0); // Always returning a distance of 0

        var data = new IrisData
        {
            X = 1,
            Y = 2,
            Z = 3,
            Classification = IrisClassification.A
        };
        var neighbors = new List<IData<IrisClassification>>
        {
            new IrisData
            {
                X = 1,
                Y = 2,
                Z = 3,
                Classification = IrisClassification.B
            }
        };

        // Act
        var result = classifier.Classify(1, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(IrisClassification.B, result);
    }

    [Fact]
    public void Classify_ShouldReturnNull_WhenTieInClassification()
    {
        // Arrange
        var classifier = new IrisClassifier();
        var mockMetric = new Mock<IMetric>();
        mockMetric
            .SetupSequence(m => m.Calculate(It.IsAny<List<double>>(), It.IsAny<List<double>>()))
            .Returns(1.0) // First neighbor
            .Returns(1.0); // Second neighbor (tie)

        var data = new IrisData
        {
            X = 4,
            Y = 5,
            Z = 6,
            Classification = IrisClassification.A
        };
        var neighbors = new List<IData<IrisClassification>>
        {
            new IrisData
            {
                X = 1,
                Y = 2,
                Z = 3,
                Classification = IrisClassification.B
            },
            new IrisData
            {
                X = 7,
                Y = 8,
                Z = 9,
                Classification = IrisClassification.C
            }
        };

        // Act
        var result = classifier.Classify(2, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(default, result);
    }

    [Fact]
    public void Classify_ShouldHandleEmptyNeighborsList()
    {
        // Arrange
        var classifier = new IrisClassifier();
        var mockMetric = new Mock<IMetric>();
        var data = new IrisData
        {
            X = 1,
            Y = 2,
            Z = 3,
            Classification = IrisClassification.A
        };
        var neighbors = new List<IData<IrisClassification>>(); // Empty list

        // Act
        var result = classifier.Classify(1, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(default, result);
    }

    [Fact]
    public void Classify_ShouldHandleSingleNeighbor_WhenKIsGreaterThanNeighborsCount()
    {
        // Arrange
        var classifier = new IrisClassifier();
        var mockMetric = new Mock<IMetric>();
        mockMetric
            .Setup(m => m.Calculate(It.IsAny<List<double>>(), It.IsAny<List<double>>()))
            .Returns(0.0);

        var data = new IrisData
        {
            X = 1,
            Y = 2,
            Z = 3,
            Classification = IrisClassification.A
        };
        var neighbors = new List<IData<IrisClassification>>
        {
            new IrisData
            {
                X = 1,
                Y = 2,
                Z = 3,
                Classification = IrisClassification.B
            }
        };

        // Act
        var result = classifier.Classify(5, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(IrisClassification.B, result);
    }

    [Fact]
    public void Classify_ShouldCorrectlySelectKClosestNeighbors()
    {
        // Arrange
        var classifier = new IrisClassifier();
        var mockMetric = new Mock<IMetric>();
        mockMetric
            .SetupSequence(m => m.Calculate(It.IsAny<List<double>>(), It.IsAny<List<double>>()))
            .Returns(10.0)
            .Returns(5.0)
            .Returns(15.0)
            .Returns(7.0)
            .Returns(3.0);

        var data = new IrisData
        {
            X = 1,
            Y = 2,
            Z = 3,
            Classification = IrisClassification.A
        };
        var neighbors = new List<IData<IrisClassification>>
        {
            new IrisData
            {
                X = 10,
                Y = 20,
                Z = 30,
                Classification = IrisClassification.B
            },
            new IrisData
            {
                X = 2,
                Y = 4,
                Z = 6,
                Classification = IrisClassification.C
            },
            new IrisData
            {
                X = 3,
                Y = 6,
                Z = 9,
                Classification = IrisClassification.B
            },
            new IrisData
            {
                X = 4,
                Y = 8,
                Z = 12,
                Classification = IrisClassification.C
            },
            new IrisData
            {
                X = 50,
                Y = 52,
                Z = 53,
                Classification = IrisClassification.A
            }
        };

        // Act
        var result = classifier.Classify(3, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(IrisClassification.C, result);
    }

    [Fact]
    public void Classify_ShouldReturnDefault_WhenKIsZero()
    {
        // Arrange
        var classifier = new IrisClassifier();
        var mockMetric = new Mock<IMetric>();
        var data = new IrisData
        {
            X = 1,
            Y = 2,
            Z = 3,
            Classification = IrisClassification.A
        };
        var neighbors = new List<IData<IrisClassification>>
        {
            new IrisData
            {
                X = 10,
                Y = 20,
                Z = 30,
                Classification = IrisClassification.B
            }
        };

        // Act
        var result = classifier.Classify(0, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(default, result);
    }
}