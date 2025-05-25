using System.Collections.Generic;
using ml_knn;
using ml_knn.Metrics;
using Moq;
using Xunit;

namespace knn.Tests;

public class ClassifierTest
{
    private enum TestClassification
    {
        A = 0, 
        B = 1,
        C = 2
    }

    private class TestData : IData<TestClassification>
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }
        public TestClassification Classification { get; set; }
    }
    
    [Fact]
    public void Classify_ShouldReturnCorrectClassification_WhenSingleNeighbor()
    {
        // Arrange
        var mockMetric = new Mock<IMetric>();
        mockMetric
            .Setup(m => m.Calculate(It.IsAny<List<double>>(), It.IsAny<List<double>>()))
            .Returns(0.0); // Always returning a distance of 0

        var data = new TestData
        {
            X = 1,
            Y = 2,
            Z = 3,
            Classification = TestClassification.A
        };
        var neighbors = new List<IData<TestClassification>>
        {
            new TestData
            {
                X = 1,
                Y = 2,
                Z = 3,
                Classification = TestClassification.B
            }
        };

        // Act
        var result = Classifier.Classify(1, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(TestClassification.B, result);
    }

    [Fact]
    public void Classify_ShouldReturnNull_WhenTieInClassification()
    {
        // Arrange
        var mockMetric = new Mock<IMetric>();
        mockMetric
            .SetupSequence(m => m.Calculate(It.IsAny<List<double>>(), It.IsAny<List<double>>()))
            .Returns(1.0) // First neighbor
            .Returns(1.0); // Second neighbor (tie)

        var data = new TestData
        {
            X = 4,
            Y = 5,
            Z = 6,
            Classification = TestClassification.A
        };
        var neighbors = new List<IData<TestClassification>>
        {
            new TestData
            {
                X = 1,
                Y = 2,
                Z = 3,
                Classification = TestClassification.B
            },
            new TestData
            {
                X = 7,
                Y = 8,
                Z = 9,
                Classification = TestClassification.C
            }
        };

        // Act
        var result = Classifier.Classify(2, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(default, result);
    }

    [Fact]
    public void Classify_ShouldHandleEmptyNeighborsList()
    {
        // Arrange
        var mockMetric = new Mock<IMetric>();
        var data = new TestData
        {
            X = 1,
            Y = 2,
            Z = 3,
            Classification = TestClassification.A
        };
        var neighbors = new List<IData<TestClassification>>(); // Empty list

        // Act
        var result = Classifier.Classify(1, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(default, result);
    }

    [Fact]
    public void Classify_ShouldHandleSingleNeighbor_WhenKIsGreaterThanNeighborsCount()
    {
        // Arrange
        var mockMetric = new Mock<IMetric>();
        mockMetric
            .Setup(m => m.Calculate(It.IsAny<List<double>>(), It.IsAny<List<double>>()))
            .Returns(0.0);

        var data = new TestData
        {
            X = 1,
            Y = 2,
            Z = 3,
            Classification = TestClassification.A
        };
        var neighbors = new List<IData<TestClassification>>
        {
            new TestData
            {
                X = 1,
                Y = 2,
                Z = 3,
                Classification = TestClassification.B
            }
        };

        // Act
        var result = Classifier.Classify(5, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(TestClassification.B, result);
    }

    [Fact]
    public void Classify_ShouldCorrectlySelectKClosestNeighbors()
    {
        // Arrange
        var mockMetric = new Mock<IMetric>();
        mockMetric
            .SetupSequence(m => m.Calculate(It.IsAny<List<double>>(), It.IsAny<List<double>>()))
            .Returns(10.0)
            .Returns(5.0)
            .Returns(15.0)
            .Returns(7.0)
            .Returns(3.0);

        var data = new TestData
        {
            X = 1,
            Y = 2,
            Z = 3,
            Classification = TestClassification.A
        };
        var neighbors = new List<IData<TestClassification>>
        {
            new TestData
            {
                X = 10,
                Y = 20,
                Z = 30,
                Classification = TestClassification.B
            },
            new TestData
            {
                X = 2,
                Y = 4,
                Z = 6,
                Classification = TestClassification.C
            },
            new TestData
            {
                X = 3,
                Y = 6,
                Z = 9,
                Classification = TestClassification.B
            },
            new TestData
            {
                X = 4,
                Y = 8,
                Z = 12,
                Classification = TestClassification.C
            },
            new TestData
            {
                X = 50,
                Y = 52,
                Z = 53,
                Classification = TestClassification.A
            }
        };

        // Act
        var result = Classifier.Classify(3, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(TestClassification.C, result!.Value);
    }

    [Fact]
    public void Classify_ShouldReturnDefault_WhenKIsZero()
    {
        // Arrange
        var mockMetric = new Mock<IMetric>();
        var data = new TestData
        {
            X = 1,
            Y = 2,
            Z = 3,
            Classification = TestClassification.A
        };
        var neighbors = new List<IData<TestClassification>>
        {
            new TestData
            {
                X = 10,
                Y = 20,
                Z = 30,
                Classification = TestClassification.B
            }
        };

        // Act
        var result = Classifier.Classify(0, data, neighbors, mockMetric.Object);

        // Assert
        Assert.Equal(default, result);
    }
}