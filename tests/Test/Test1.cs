using Microsoft.VisualStudio.TestTools.UnitTesting;
using Image_Editor_lib;
using System.IO;
namespace ImageEditorLib.Tests
{
    internal static class TestImageFactory
    {
        public static byte[,] Gradient(int height, int width)
        {
            byte[,] image = new byte[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    image[i, j] = (byte)((i * 7 + j * 3) % 256);
                }
            }
            return image;
        }

        public static byte[,] Constant(int height, int width, byte value)
        {
            byte[,] image = new byte[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    image[i, j] = value;
                }
            }
            return image;
        }

        public static string CreateTempFolder()
        {
            string dir = Path.Combine(Path.GetTempPath(), "ImageEditor_Tests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            return dir;
        }

        public static string SaveTempImage(string dir, string filename, byte[,] image)
        {
            string path = Path.Combine(dir, filename);
            IOImage.Save2DbyteArrayAsImage(image, path);
            return path;
        }
    }

    [TestClass]
    public class MyImageTests
    {
        [TestMethod]
        public void Constructor_SetsProperties_Correctly()
        {
            int width = 100;
            int height = 200;
            byte[] data = new byte[100 * 200];
            string name = "test.png";
            var image = new myImage(width, height, data, name);
            Assert.AreEqual(width, image.Width);
            Assert.AreEqual(height, image.Height);
            Assert.AreEqual(data, image.Data);
            Assert.AreEqual(name, image.Name);
        }
    }
    [TestClass]
    public class KernelsTests
    {
        [TestMethod]
        public void GaussianBlurKernel_HasCorrectSize()
        {
            Assert.AreEqual(5, Kernels.gaussianBlurKernel.GetLength(0));
            Assert.AreEqual(5, Kernels.gaussianBlurKernel.GetLength(1));
        }
        [TestMethod]
        public void GaussianBlurKernel_SumEqualsOne()
        {
            float sum = 0;
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    sum += Kernels.gaussianBlurKernel[i, j];
                }
            }
            Assert.AreEqual(1f, sum, 0.001f);
        }
        [TestMethod]
        public void EdgeKernel_HasCorrectSize()
        {
            Assert.AreEqual(5, Kernels.egdeKernel.GetLength(0));
            Assert.AreEqual(5, Kernels.egdeKernel.GetLength(1));
        }
        [TestMethod]
        public void ScaryKernel_HasCorrectSize()
        {
            Assert.AreEqual(3, Kernels.scaryKernel.GetLength(0));
            Assert.AreEqual(3, Kernels.scaryKernel.GetLength(1));
        }
        [TestMethod]
        public void EdgeKernel_SumEqualsZero()
        {
            float sum = 0;
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    sum += Kernels.egdeKernel[i, j];
                }
            }
            Assert.AreEqual(0f, sum, 0.001f);
        }
        [TestMethod]
        public void ScaryKernel_SumEqualsZero()
        {
            float sum = 0;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    sum += Kernels.scaryKernel[i, j];
                }
            }
            Assert.AreEqual(0f, sum, 0.001f);
        }
    }
    [TestClass]
    public class ConvolutionFunctionsDataParallelTests
    {
        private const int ProcessorCount = 4;

        [TestMethod]
        public void ApplyFilterDataParallel_ReturnsCorrectDimensions()
        {
            byte[,] image = new byte[5, 5];
            float[,] filter = new float[,]
            {
                { 0, 0, 0 },
                { 0, 1, 0 },
                { 0, 0, 0 }
            };
            var result = ConvolutionFunctions.applyFilterDataParallel(filter, image, ProcessorCount);
            Assert.AreEqual(5, result.GetLength(0));
            Assert.AreEqual(5, result.GetLength(1));
        }
        [TestMethod]
        public void ApplyFilterDataParallel_IdentityKernel_ReturnsOriginalImage()
        {
            byte[,] image = new byte[5, 5]
            {
                { 10, 20, 30, 40, 50 },
                { 60, 70, 80, 90, 100 },
                { 110, 120, 130, 140, 150 },
                { 160, 170, 180, 190, 200 },
                { 210, 220, 230, 240, 250 }
            };
            float[,] identity = new float[,]
            {
                { 0, 0, 0 },
                { 0, 1, 0 },
                { 0, 0, 0 }
            };
            var result = ConvolutionFunctions.applyFilterDataParallel(identity, image, ProcessorCount);
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    Assert.AreEqual(image[i, j], result[i, j]);
                }
            }
        }
        [TestMethod]
        public void ApplyFilterDataParallel_GaussianBlur_CenterPixelHandComputed()
        {
            byte[,] image = new byte[3, 3]
            {
                { 0, 0, 0 },
                { 0, 255, 0 },
                { 0, 0, 0 }
            };
            var result = ConvolutionFunctions.applyFilterDataParallel(Kernels.gaussianBlurKernel, image, ProcessorCount);
            Assert.AreEqual(95, result[1, 1]);
        }
        [TestMethod]
        public void ApplyFilterDataParallel_ConstantImage_UnchangedByGaussian()
        {
            byte[,] image = TestImageFactory.Constant(9, 9, 128);
            var result = ConvolutionFunctions.applyFilterDataParallel(Kernels.gaussianBlurKernel, image, ProcessorCount);
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    Assert.AreEqual(128, result[i, j], $"Mismatch at ({i},{j})");
                }
            }
        }
        [TestMethod]
        public void ApplyFilterDataParallel_ConstantImage_ZeroedByZeroSumKernel()
        {
            byte[,] image = TestImageFactory.Constant(9, 9, 128);
            var edge = ConvolutionFunctions.applyFilterDataParallel(Kernels.egdeKernel, image, ProcessorCount);
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    Assert.AreEqual(0, edge[i, j], $"Mismatch at ({i},{j})");
                }
            }
        }
        [TestMethod]
        public void ApplyFilterDataParallel_EdgeKernel_DetectsEdges()
        {
            byte[,] image = new byte[5, 5]
            {
                { 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0 },
                { 0, 0, 255, 0, 0 },
                { 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0 }
            };
            var result = ConvolutionFunctions.applyFilterDataParallel(Kernels.egdeKernel, image, ProcessorCount);
            Assert.AreNotEqual(0, result[2, 2]);
        }
        [TestMethod]
        public void ApplyFilterDataParallel_ClampsValuesAbove255()
        {
            byte[,] image = new byte[3, 3]
            {
                { 0, 0, 0 },
                { 0, 200, 0 },
                { 0, 0, 0 }
            };
            float[,] amp = new float[,]
            {
                { 0, 0, 0 },
                { 0, 2, 0 },
                { 0, 0, 0 }
            };
            var result = ConvolutionFunctions.applyFilterDataParallel(amp, image, ProcessorCount);
            Assert.AreEqual(255, result[1, 1]);
            Assert.AreEqual(0, result[0, 0]);
        }
        [TestMethod]
        public void ApplyFilterDataParallel_ClampsNegativeToZero()
        {
            byte[,] image = new byte[3, 3]
            {
                { 0, 0, 0 },
                { 0, 100, 0 },
                { 0, 0, 0 }
            };
            float[,] neg = new float[,]
            {
                { 0, 0, 0 },
                { 0, -2, 0 },
                { 0, 0, 0 }
            };
            var result = ConvolutionFunctions.applyFilterDataParallel(neg, image, ProcessorCount);
            Assert.AreEqual(0, result[1, 1]);
        }
    }
    [TestClass]
    public class ProcessByDataTests
    {
        private const int ProcessorCount = 4;

        [TestMethod]
        public void ProcessByData_ReturnsZeroAndCreatesExpectedOutput()
        {
            string dir = TestImageFactory.CreateTempFolder();
            try
            {
                byte[,] image = TestImageFactory.Gradient(37, 41);
                string input = TestImageFactory.SaveTempImage(dir, "input.png", image);
                var filters = new List<float[,]> { Kernels.gaussianBlurKernel };

                int code = ConvolutionFunctions.ProcessByData(new[] { input }, filters, ProcessorCount);

                Assert.AreEqual(0, code);
                var actual = IOImage.LoadAs2DArray(Path.Combine(dir, "input_out.png"));
                var expected = ConvolutionFunctions.applyFilterDataParallel(Kernels.gaussianBlurKernel, image, ProcessorCount);
                AssertSameImage(expected, actual);
            }
            finally
            {
                Directory.Delete(dir, true);
            }
        }
        [TestMethod]
        public void ProcessByData_ReturnsMinusOneOnMissingFile()
        {
            var filters = new List<float[,]> { Kernels.gaussianBlurKernel };
            int code = ConvolutionFunctions.ProcessByData(new[] { "nonexistent_file.png" }, filters, ProcessorCount);
            Assert.AreEqual(-1, code);
        }

        internal static void AssertSameImage(byte[,] expected, byte[,] actual)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    Assert.AreEqual(expected[i, j], actual[i, j], $"Mismatch at ({i},{j})");
                }
            }
        }
    }
    [TestClass]
    public class ProcessByTasksTests
    {
        private const int AgentCount = 4;

        [TestMethod]
        public void ProcessByTasks_ReturnsZeroAndCreatesExpectedOutput()
        {
            string dir = TestImageFactory.CreateTempFolder();
            try
            {
                byte[,] image = TestImageFactory.Gradient(37, 41);
                string input = TestImageFactory.SaveTempImage(dir, "input.png", image);
                var filters = new List<float[,]> { Kernels.gaussianBlurKernel };

                int code = ConvolutionFunctions.ProcessByTasks(new[] { input }, filters, AgentCount);

                Assert.AreEqual(0, code);
                var actual = IOImage.LoadAs2DArray(Path.Combine(dir, "input_out.png"));
                var expected = ConvolutionFunctions.applyFilterDataParallel(Kernels.gaussianBlurKernel, image, AgentCount);
                ProcessByDataTests.AssertSameImage(expected, actual);
            }
            finally
            {
                Directory.Delete(dir, true);
            }
        }
        [TestMethod]
        public void ProcessByTasks_ReturnsMinusOneOnMissingFile()
        {
            var filters = new List<float[,]> { Kernels.gaussianBlurKernel };
            int code = ConvolutionFunctions.ProcessByTasks(new[] { "nonexistent_file.png" }, filters, AgentCount);
            Assert.AreEqual(-1, code);
        }
    }
    [TestClass]
    public class ParallelismStrategiesCorrespondenceTests
    {
        private const int Count = 4;

        [TestMethod]
        public void SingleImage_DataAndTaskStrategies_ProduceIdenticalImages()
        {
            string dir = TestImageFactory.CreateTempFolder();
            try
            {
                byte[,] source = TestImageFactory.Gradient(37, 41);
                string dataInput = TestImageFactory.SaveTempImage(dir, "data.png", source);
                string taskInput = TestImageFactory.SaveTempImage(dir, "tasks.png", source);
                var filters = new List<float[,]> { Kernels.gaussianBlurKernel };

                int dataCode = ConvolutionFunctions.ProcessByData(new[] { dataInput }, filters, Count);
                int taskCode = ConvolutionFunctions.ProcessByTasks(new[] { taskInput }, filters, Count);

                Assert.AreEqual(0, dataCode);
                Assert.AreEqual(0, taskCode);

                var expected = ConvolutionFunctions.applyFilterDataParallel(Kernels.gaussianBlurKernel, source, Count);
                var dataOut = IOImage.LoadAs2DArray(Path.Combine(dir, "data_out.png"));
                var taskOut = IOImage.LoadAs2DArray(Path.Combine(dir, "tasks_out.png"));

                for (int i = 0; i < source.GetLength(0); i++)
                {
                    for (int j = 0; j < source.GetLength(1); j++)
                    {
                        Assert.AreEqual(expected[i, j], dataOut[i, j], $"Data mismatch at ({i},{j})");
                        Assert.AreEqual(expected[i, j], taskOut[i, j], $"Task mismatch at ({i},{j})");
                    }
                }
            }
            finally
            {
                Directory.Delete(dir, true);
            }
        }
        [TestMethod]
        public void MultipleImages_PerFilterPairing_DataAndTaskAgree()
        {
            string dir = TestImageFactory.CreateTempFolder();
            try
            {
                byte[,] imageA = TestImageFactory.Gradient(24, 29);
                byte[,] imageB = TestImageFactory.Gradient(17, 31);
                var filters = new List<float[,]> { Kernels.gaussianBlurKernel, Kernels.scaryKernel };

                string dataInA = TestImageFactory.SaveTempImage(dir, "data_a.png", imageA);
                string dataInB = TestImageFactory.SaveTempImage(dir, "data_b.png", imageB);
                string taskInA = TestImageFactory.SaveTempImage(dir, "task_a.png", imageA);
                string taskInB = TestImageFactory.SaveTempImage(dir, "task_b.png", imageB);

                int dataCode = ConvolutionFunctions.ProcessByData(new[] { dataInA, dataInB }, filters, Count);
                int taskCode = ConvolutionFunctions.ProcessByTasks(new[] { taskInA, taskInB }, filters, Count);
                Assert.AreEqual(0, dataCode);
                Assert.AreEqual(0, taskCode);

                var expectedA = ConvolutionFunctions.applyFilterDataParallel(filters[0], imageA, Count);
                var expectedB = ConvolutionFunctions.applyFilterDataParallel(filters[1], imageB, Count);
                var dataOutA = IOImage.LoadAs2DArray(Path.Combine(dir, "data_a_out.png"));
                var dataOutB = IOImage.LoadAs2DArray(Path.Combine(dir, "data_b_out.png"));
                var taskOutA = IOImage.LoadAs2DArray(Path.Combine(dir, "task_a_out.png"));
                var taskOutB = IOImage.LoadAs2DArray(Path.Combine(dir, "task_b_out.png"));

                ProcessByDataTests.AssertSameImage(expectedA, dataOutA);
                ProcessByDataTests.AssertSameImage(expectedB, dataOutB);
                ProcessByDataTests.AssertSameImage(expectedA, taskOutA);
                ProcessByDataTests.AssertSameImage(expectedB, taskOutB);
            }
            finally
            {
                Directory.Delete(dir, true);
            }
        }
        [TestMethod]
        public void FewerFiltersThanImages_LastFilterRepeated_DataAndTaskAgree()
        {
            string dir = TestImageFactory.CreateTempFolder();
            try
            {
                byte[,] image1 = TestImageFactory.Gradient(20, 20);
                byte[,] image2 = TestImageFactory.Gradient(21, 22);
                byte[,] image3 = TestImageFactory.Gradient(23, 24);
                var filters = new List<float[,]> { Kernels.scaryKernel };

                string[] dataInputs =
                {
                    TestImageFactory.SaveTempImage(dir, "data_1.png", image1),
                    TestImageFactory.SaveTempImage(dir, "data_2.png", image2),
                    TestImageFactory.SaveTempImage(dir, "data_3.png", image3)
                };
                string[] taskInputs =
                {
                    TestImageFactory.SaveTempImage(dir, "task_1.png", image1),
                    TestImageFactory.SaveTempImage(dir, "task_2.png", image2),
                    TestImageFactory.SaveTempImage(dir, "task_3.png", image3)
                };

                Assert.AreEqual(0, ConvolutionFunctions.ProcessByData(dataInputs, filters, Count));
                Assert.AreEqual(0, ConvolutionFunctions.ProcessByTasks(taskInputs, filters, Count));

                byte[][,] sources = new byte[][,] { image1, image2, image3 };
                for (int k = 0; k < 3; k++)
                {
                    var expected = ConvolutionFunctions.applyFilterDataParallel(filters[0], sources[k], Count);
                    var dataOut = IOImage.LoadAs2DArray(Path.Combine(dir, $"data_{k + 1}_out.png"));
                    var taskOut = IOImage.LoadAs2DArray(Path.Combine(dir, $"task_{k + 1}_out.png"));
                    ProcessByDataTests.AssertSameImage(expected, dataOut);
                    ProcessByDataTests.AssertSameImage(expected, taskOut);
                }
            }
            finally
            {
                Directory.Delete(dir, true);
            }
        }
    }
    [TestClass]
    public class IOImageTests
    {
        [TestMethod]
        public void LoadAs2DArray_ThrowsOnNonExistentFile()
        {
            Assert.ThrowsException<FileNotFoundException>(
                () => IOImage.LoadAs2DArray("nonexistent_file.png")
            );
        }
        [TestMethod]
        public void LoadAsImage_ThrowsOnNonExistentFile()
        {
            Assert.ThrowsException<FileNotFoundException>(
                () => IOImage.LoadAsImage("nonexistent_file.png")
            );
        }
        [TestMethod]
        public void Save2DbyteArrayAsImage_CreatesFile()
        {
            byte[,] data = new byte[10, 10];
            string tempPath = Path.Combine(Path.GetTempPath(), "test_output.png");
            try
            {
                IOImage.Save2DbyteArrayAsImage(data, tempPath);
                Assert.IsTrue(File.Exists(tempPath));
            }
            finally
            {
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }
        [TestMethod]
        public void LoadAsImage_Equal_LoadAs2DArray()
        {
            string path = Path.Combine(Path.GetTempPath(), "roundtrip_test.png");
            byte[,] data = new byte[4, 5]
            {
                { 10, 20, 30, 40, 50 },
                { 60, 70, 80, 90, 100 },
                { 110, 120, 130, 140, 150 },
                { 160, 170, 180, 190, 200 },
            };
            try
            {
                IOImage.Save2DbyteArrayAsImage(data, path);
                var resultArray = IOImage.LoadAs2DArray(path);
                var resultImage = IOImage.LoadAsImage(path);
                Assert.AreEqual(resultImage.Height, resultArray.GetLength(0));
                Assert.AreEqual(resultImage.Width, resultArray.GetLength(1));
                for (int i = 0; i < resultArray.GetLength(0); i++)
                {
                    for (int j = 0; j < resultArray.GetLength(1); j++)
                    {
                        Assert.AreEqual(resultArray[i, j], resultImage.Data[i * resultArray.GetLength(1) + j]);
                    }
                }
            }
            finally
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
        }
    }
}