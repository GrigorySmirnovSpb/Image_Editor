using Microsoft.VisualStudio.TestTools.UnitTesting;
using Image_Editor_lib;
using System.IO;
namespace ImageEditorLib.Tests
{
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
        [TestMethod]
        public void Data_DefaultIsNull()
        {
            var image = new myImage(10, 10, null, "test");
            Assert.IsNull(image.Data);
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
    }
    [TestClass]
    public class ConvolutionFunctionsTests
    {
        [TestMethod]
        public void ApplyFilter_ReturnsCorrectDimensions()
        {
            byte[,] image = new byte[5, 5];
            float[,] filter = new float[,]
            {
                { 0, 0, 0 },
                { 0, 1, 0 },
                { 0, 0, 0 }
            };
            var result = ConvolutionFunctions.applyFilter(filter, image);
            Assert.AreEqual(5, result.GetLength(0));
            Assert.AreEqual(5, result.GetLength(1));
        }
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
            float[,] identity = new float[,]
            {
                { 0, 0, 0 },
                { 0, 1, 0 },
                { 0, 0, 0 }
            };
            var result = ConvolutionFunctions.applyFilter(identity, image);
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    Assert.AreEqual(image[i, j], result[i, j]);
                }
            }
        }
        [TestMethod]
        public void ApplyFilter_GaussianBlur_MakesImageSmoother()
        {
            byte[,] image = new byte[3, 3]
            {
                { 0, 0, 0 },
                { 0, 255, 0 },
                { 0, 0, 0 }
            };
            var result = ConvolutionFunctions.applyFilter(Kernels.gaussianBlurKernel, image);
            Assert.IsTrue(result[1, 1] < 255);
        }
        [TestMethod]
        public void ApplyFilter_EdgeKernel_DetectsEdges()
        {
            byte[,] image = new byte[5, 5]
            {
                { 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0 },
                { 0, 0, 255, 0, 0 },
                { 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0 }
            };
            var result = ConvolutionFunctions.applyFilter(Kernels.egdeKernel, image);
            Assert.AreNotEqual(0, result[2, 2]);
        }
        [TestMethod]
        public void ApplyFilter_SmallKernelOnLargeImage_Works()
        {
            byte[,] image = new byte[100, 100];
            float[,] filter = new float[,]
            {
                { 1, 2, 1 },
                { 2, 4, 2 },
                { 1, 2, 1 }
            };
            var result = ConvolutionFunctions.applyFilter(filter, image);
            Assert.AreEqual(100, result.GetLength(0));
            Assert.AreEqual(100, result.GetLength(1));
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
    }
}