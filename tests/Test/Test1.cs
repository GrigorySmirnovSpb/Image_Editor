using Image_Editor_lib;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;

namespace ImageEditorLib.Tests
{
    internal static class TestAccelerator
    {
        private static readonly object Sync = new();
        private static Context? _context;
        private static Accelerator? _accelerator;

        public static Accelerator Create()
        {
            lock (Sync)
            {
                if (_accelerator is not null)
                    return _accelerator;

                _context ??= Context.CreateDefault();
                Device device = _context.Devices
                    .FirstOrDefault(d => d.AcceleratorType == AcceleratorType.OpenCL)
                    ?? _context.Devices[0];
                _accelerator = device.CreateAccelerator(_context);
                return _accelerator;
            }
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
            for (int i = 0; i < Kernels.gaussianBlurKernel.GetLength(0); i++)
                for (int j = 0; j < Kernels.gaussianBlurKernel.GetLength(1); j++)
                    sum += Kernels.gaussianBlurKernel[i, j];
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
    }

    [TestClass]
    public class ConvolutionFunctionsTests
    {
        private static readonly Accelerator Accelerator = TestAccelerator.Create();

        private static byte[,] ApplyFilter(float[,] filter, byte[,] image) =>
            ConvolutionFunctions.applyFilter(filter, image, Accelerator);

        private static float[,] Identity3x3() => new float[,]
        {
            { 0, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 0 }
        };

        [TestMethod]
        public void ApplyFilter_IdentityKernel_ReturnsOriginalImage()
        {
            byte[,] image = new byte[5, 5]
            {
                { 10, 20, 30, 40, 50 },
                { 60, 70, 80, 90, 100 },
                { 110, 120, 130, 140, 150 },
                { 160, 170, 180, 190, 200 },
                { 210, 220, 230, 240, 250 }
            };
            var result = ApplyFilter(Identity3x3(), image);
            for (int i = 0; i < image.GetLength(0); i++)
                for (int j = 0; j < image.GetLength(1); j++)
                    Assert.AreEqual(image[i, j], result[i, j]);
        }

        [TestMethod]
        public void ApplyFilter_Identity_PreservesImpulsePosition_OnNonSquareImage()
        {
            byte[,] image = new byte[6, 10];
            image[3, 7] = 255;

            var result = ApplyFilter(Identity3x3(), image);

            double total = 0;
            for (int i = 0; i < result.GetLength(0); i++)
                for (int j = 0; j < result.GetLength(1); j++)
                    total += result[i, j];

            Assert.AreEqual(255, result[3, 7]);
            Assert.AreEqual(255.0, total);
        }

        [TestMethod]
        public void ApplyFilter_GaussianBlur_BlursImpulse()
        {
            byte[,] image = new byte[7, 7];
            image[3, 3] = 255;

            var result = ApplyFilter(Kernels.gaussianBlurKernel, image);

            Assert.IsTrue(result[3, 3] < 255, "Center should lose brightness");
            Assert.IsTrue(result[3, 3] > 0, "Center should keep some brightness");
            Assert.IsTrue(result[3, 4] > 0, "Neighbor should receive brightness");
            Assert.IsTrue(result[1, 3] > 0, "Far neighbor should receive brightness");
        }

        [TestMethod]
        public void ApplyFilter_EdgeKernel_DetectsEdges()
        {
            byte[,] image = new byte[5, 5];
            image[2, 2] = 255;

            var result = ApplyFilter(Kernels.egdeKernel, image);

            Assert.AreEqual(255, result[2, 2]);
        }

        [TestMethod]
        public void ApplyFilter_ClampsTo255_HighWeights()
        {
            byte[,] image = new byte[5, 5];
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    image[i, j] = 255;

            float[,] heavy = new float[,]
            {
                { 10, 10, 10 },
                { 10, 10, 10 },
                { 10, 10, 10 }
            };
            var result = ApplyFilter(heavy, image);

            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    Assert.AreEqual(255, result[i, j], "Result must not overflow");
        }

        [TestMethod]
        public void ApplyFilter_ClampsTo0_NegativeWeights()
        {
            byte[,] image = new byte[5, 5];
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    image[i, j] = 255;

            float[,] negative = new float[,]
            {
                { -1, -1, -1 },
                { -1, -1, -1 },
                { -1, -1, -1 }
            };
            var result = ApplyFilter(negative, image);

            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    Assert.AreEqual(0, result[i, j], "Result must not underflow");
        }

        [TestMethod]
        public void ApplyFilter_LargeImage_ReturnsCorrectDimensions()
        {
            byte[,] image = new byte[100, 60];
            var result = ApplyFilter(Kernels.scaryKernel, image);
            Assert.AreEqual(100, result.GetLength(0));
            Assert.AreEqual(60, result.GetLength(1));
        }
    }

    [TestClass]
    public class ProcessByTasksTests
    {
        private static readonly Accelerator Accelerator = TestAccelerator.Create();

        private static float[,] Identity3x3() => new float[,]
        {
            { 0, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 0 }
        };

        private static byte[,] ConstantImage(int height, int width, byte value)
        {
            byte[,] image = new byte[height, width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    image[i, j] = value;
            return image;
        }

        private static string CreateTempImage(byte[,] data)
        {
            string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + ".png");
            IOImage.Save2DbyteArrayAsImage(data, path);
            return path;
        }

        private static string OutputPathFor(string input) => Path.Combine(
            Path.GetDirectoryName(input) ?? ".",
            Path.GetFileNameWithoutExtension(input) + "_out.png");

        private static void AssertOutputEquals(string outputPath, byte[,] expected)
        {
            Assert.IsTrue(File.Exists(outputPath), $"Output file not created: {outputPath}");
            byte[,] loaded = IOImage.LoadAs2DArray(outputPath);
            Assert.AreEqual(expected.GetLength(0), loaded.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), loaded.GetLength(1));
            for (int i = 0; i < expected.GetLength(0); i++)
                for (int j = 0; j < expected.GetLength(1); j++)
                    Assert.AreEqual(expected[i, j], loaded[i, j]);
        }

        [TestMethod]
        public void ProcessByTasks_ProcessesAllFiles_CreatesAllOutputs()
        {
            System.Collections.Generic.List<float[,]> filters = new() { Identity3x3() };
            byte[,] expected = ConstantImage(4, 6, 200);

            string[] files = new string[4];
            try
            {
                for (int i = 0; i < files.Length; i++)
                    files[i] = CreateTempImage(expected);

                int result = ConvolutionFunctions.ProcessByTasks(files, filters, 2);

                Assert.AreEqual(0, result);
                foreach (string file in files)
                    AssertOutputEquals(OutputPathFor(file), expected);
            }
            finally
            {
                foreach (string file in files)
                {
                    if (File.Exists(file)) File.Delete(file);
                    string outName = OutputPathFor(file);
                    if (File.Exists(outName)) File.Delete(outName);
                }
            }
        }

        [TestMethod]
        public void ProcessByTasks_FileCountExceedsAgents_StillProcessesAll()
        {
            System.Collections.Generic.List<float[,]> filters = new() { Identity3x3() };
            byte[,] expected = ConstantImage(5, 5, 128);

            string[] files = new string[3];
            try
            {
                for (int i = 0; i < files.Length; i++)
                    files[i] = CreateTempImage(expected);

                int result = ConvolutionFunctions.ProcessByTasks(files, filters, 1);

                Assert.AreEqual(0, result);
                foreach (string file in files)
                    AssertOutputEquals(OutputPathFor(file), expected);
                foreach (string file in files)
                    Assert.IsTrue(File.Exists(OutputPathFor(file)));
            }
            finally
            {
                foreach (string file in files)
                {
                    if (File.Exists(file)) File.Delete(file);
                    string outName = OutputPathFor(file);
                    if (File.Exists(outName)) File.Delete(outName);
                }
            }
        }

        [TestMethod]
        public void ProcessByTasks_ResultMatchesSequentialApplyFilter()
        {
            System.Collections.Generic.List<float[,]> filters = new() { Kernels.gaussianBlurKernel };
            byte[,] image = new byte[7, 9]
            {
                { 10, 20, 30, 40, 50, 60, 70, 80, 90 },
                { 90, 80, 70, 60, 50, 40, 30, 20, 10 },
                { 15, 25, 35, 45, 55, 65, 75, 85, 95 },
                { 5, 15, 25, 35, 45, 55, 65, 75, 85 },
                { 24, 32, 40, 48, 56, 64, 72, 80, 88 },
                { 200, 180, 160, 140, 120, 100, 80, 60, 40 },
                { 1, 3, 5, 7, 9, 11, 13, 99, 255 }
            };

            byte[,] expected = ConvolutionFunctions.applyFilter(Kernels.gaussianBlurKernel, image, Accelerator);

            string file = CreateTempImage(image);
            try
            {
                int result = ConvolutionFunctions.ProcessByTasks(new[] { file }, filters, 1);

                Assert.AreEqual(0, result);
                AssertOutputEquals(OutputPathFor(file), expected);
            }
            finally
            {
                if (File.Exists(file)) File.Delete(file);
                string outName = OutputPathFor(file);
                if (File.Exists(outName)) File.Delete(outName);
            }
        }

        [TestMethod]
        public void ProcessByTasks_ReturnsMinusOne_OnMissingFile()
        {
            System.Collections.Generic.List<float[,]> filters = new() { Identity3x3() };

            int result = ConvolutionFunctions.ProcessByTasks(
                new[] { Path.Combine(Path.GetTempPath(), "nonexistent_for_tests.png") },
                filters, 1);

            Assert.AreEqual(-1, result);
        }
    }

    [TestClass]
    public class IOImageTests
    {
        [TestMethod]
        public void LoadAs2DArray_ThrowsOnNonExistentFile()
        {
            Assert.ThrowsException<FileNotFoundException>(
                () => IOImage.LoadAs2DArray("nonexistent_file.png"));
        }

        [TestMethod]
        public void SaveThenLoad_RoundtripsPixels_OnNonSquareImage()
        {
            byte[,] data = new byte[4, 6]
            {
                { 10, 20, 30, 40, 50, 60 },
                { 70, 80, 90, 100, 110, 120 },
                { 130, 140, 150, 60, 170, 180 },
                { 190, 200, 210, 220, 230, 240 }
            };
            string tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + ".png");
            try
            {
                IOImage.Save2DbyteArrayAsImage(data, tempPath);
                byte[,] loaded = IOImage.LoadAs2DArray(tempPath);

                Assert.AreEqual(4, loaded.GetLength(0));
                Assert.AreEqual(6, loaded.GetLength(1));
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 6; j++)
                        Assert.AreEqual(data[i, j], loaded[i, j]);
            }
            finally
            {
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }
    }
}