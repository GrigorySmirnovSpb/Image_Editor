namespace Image_Editor_lib;
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Xml.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using ILGPU;
using ILGPU.Runtime;


public class IOImage
{
    public static byte[,] LoadAs2DArray (string filename)
    {
        var img = Image.Load<L8> (filename);
        byte[,] result = new byte[img.Height, img.Width];

        for (int i = 0; i < img.Height; i++)
        {
            for (int j = 0; j < img.Width; j++)
            {
                result[i,j] = img[j,i].PackedValue;
            }
        }
        return result;
    }

    public static void Save2DbyteArrayAsImage(byte [,] imagedata, string filename)
    {
        var h = imagedata.GetLength(0);
        var w = imagedata.GetLength(1);

        byte[] flatArray = new byte[w * h];
        int index = 0;
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                flatArray[index++] = imagedata[i, j];
            }
        }
        var img = Image.LoadPixelData<L8>(flatArray, w, h);
        img.Save(filename);
    }
}
public class Kernels
{
    public static float[,] gaussianBlurKernel = new float[,]
    {
        { 1f/256f, 4f/256f,  6f/256f,  4f/256f,  1f/256f },
        { 4f/256f, 16f/256f, 24f/256f, 16f/256f, 4f/256f },
        { 6f/256f, 24f/256f, 36f/256f, 24f/256f, 6f/256f },
        { 4f/256f, 16f/256f, 24f/256f, 16f/256f, 4f/256f },
        { 1f/256f, 4f/256f,  6f/256f,  4f/256f,  1f/256f }
    };
    public static float[,] egdeKernel = new float[,]
    {
        { 0f, 0f, -1f, 0f, 0f },
        { 0f, 0f, -1f, 0f, 0f },
        { 0f, 0f,  2f, 0f, 0f },
        { 0f, 0f,  0f, 0f, 0f },
        { 0f, 0f,  0f, 0f, 0f }
    };
    public static float[,] scaryKernel = new float[,]
    {
        { 1f, 0f, -1f},
        { 2f, 0f, -2f},
        { 1f, 0f, -1f}
    };
}
public class ConvolutionFunctions
{
    public static byte[,] applyFilter(float[,] filter, byte[,] image, Accelerator accelerator) 
    {
        var imgH = image.GetLength(0);
        var imgW = image.GetLength(1);

        var filterD = filter.GetLength(0) / 2;
        var filterSize = filter.GetLength(0);

        float[] source = new float[imgH * imgW];

        for (int y = 0; y < imgH; y++)
        {
            for (int x = 0; x < imgW; x++)
            {
                source[y * imgW + x] = image[y, x];
            }
        }
        
        float[] flatFilter = new float[filterSize * filterSize];
        Buffer.BlockCopy(filter, 0, flatFilter, 0, flatFilter.Length * sizeof(float));

        using var buffer = accelerator.Allocate1D<float>(source.Length);
        using var filterBuffer = accelerator.Allocate1D<float>(flatFilter.Length);
        using var resultBuffer = accelerator.Allocate1D<float>(source.Length);
        buffer.CopyFromCPU(source);
        filterBuffer.CopyFromCPU(flatFilter);

        var kernel = accelerator.LoadAutoGroupedKernel<
            Action<AcceleratorStream, Index2D, ArrayView<float>, ArrayView<float>, int, int, ArrayView<float>, int>,
            Action<Index2D, ArrayView<float>, ArrayView<float>, int, int, ArrayView<float>, int>>(processPixelGPU);
        kernel(accelerator.DefaultStream, new Index2D(imgW, imgH), buffer.View, resultBuffer.View, imgH, imgW, filterBuffer.View, filterD);

        accelerator.Synchronize();

        float[] dst = new float[source.Length];
        resultBuffer.CopyToCPU(dst);

        byte[,] result = new byte[imgH, imgW];
        for (int y = 0; y < imgH; y++)
        {
            for (int x = 0; x < imgW; x++)
            {
                result[y, x] = (byte)Math.Clamp(dst[y * imgW + x], 0, 255);
            }
        }
        return result;
    }

    private static void processPixelGPU(Index2D index, ArrayView<float> input, ArrayView<float> output, int imgH, int imgW, ArrayView<float> filter, int filterD)
    {
        var px = index.Y;
        var py = index.X;
        var filterSize = filterD * 2 + 1;

        float sum = 0;
        for (int i = px - filterD; i <= px + filterD; i++)
        {
            for (int j = py - filterD; j <= py + filterD; j++)
            {
                var flatF = (i - px + filterD) * filterSize + (j - py + filterD);
                if (i < 0 || i >= imgH || j < 0 || j >= imgW)
                {
                    sum += input[px * imgW + py] * filter[flatF];
                }
                else
                {
                    sum += input[i * imgW + j] * filter[flatF];
                }
            }
        }
        output[px * imgW + py] = sum < 0f ? 0f : (sum > 255f ? 255f : sum);
    }

    public static int ProcessByTasks(string[] files, List<float[,]> filters, int agentCount)
    {
        int n = files.Length;
        int readerCount = Math.Max(1, agentCount / 2);
        int writerCount = Math.Max(1, agentCount / 2);

        var pendingFiles = new ConcurrentQueue<int>(Enumerable.Range(0, n));
        var toProcess = new BlockingCollection<(byte[,] Image, float[,] Filter, string OutName)>(agentCount * 2);
        var toWrite = new BlockingCollection<(byte[,] Result, string OutName)>(agentCount * 2);

        var workers = new Task[agentCount];
        try
        {
            Accelerator[] accelerators = new Accelerator[agentCount];
            Context[] contexts = new Context[agentCount];
            try
            {
                for (int a = 0; a < agentCount; a++)
                {
                    var context = Context.CreateDefault();
                    Device device = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.OpenCL) ?? context.Devices[0];
                    contexts[a] = context;
                    accelerators[a] = device.CreateAccelerator(context);

                    applyFilter(Kernels.gaussianBlurKernel, new byte[8, 8], accelerators[a]);
                }

                var readers = new Task[readerCount];
                for (int r = 0; r < readerCount; r++)
                {
                    int local_r = r;
                    readers[local_r] = Task.Run(() =>
                    {
                        while (pendingFiles.TryDequeue(out int i))
                        {
                            byte[,] image = IOImage.LoadAs2DArray(files[i]);
                            int filterIndex = Math.Min(i, filters.Count - 1);
                            string outName = Path.Combine(
                                Path.GetDirectoryName(files[i]) ?? ".",
                                Path.GetFileNameWithoutExtension(files[i]) + "_out.png");
                            toProcess.Add((image, filters[filterIndex], outName));
                        }
                    });
                }

                for (int w = 0; w < agentCount; w++)
                {
                    int local_w = w;
                    workers[local_w] = Task.Run(() =>
                    {
                        foreach ((byte[,] Image, float[,] Filter, string OutName) item in toProcess.GetConsumingEnumerable())
                        {
                            byte[,] result = applyFilter(item.Filter, item.Image, accelerators[local_w]);
                            toWrite.Add((result, item.OutName));
                        }
                    });
                }

                var writers = new Task[writerCount];
                for (int w = 0; w < writerCount; w++)
                {
                    int local_w = w;
                    writers[local_w] = Task.Run(() =>
                    {
                        foreach ((byte[,] Result, string OutName) item in toWrite.GetConsumingEnumerable())
                        {
                            IOImage.Save2DbyteArrayAsImage(item.Result, item.OutName);
                            Console.WriteLine($"Saved: {item.OutName}");
                        }
                    });
                }

                try 
                { 
                    Task.WaitAll(readers); 
                }
                finally 
                { 
                    toProcess.CompleteAdding(); 
                }
                try 
                { 
                    Task.WaitAll(workers); 
                }
                finally 
                { 
                    toWrite.CompleteAdding(); 
                }
                    Task.WaitAll(writers);
            }
            finally
            {
                for (int a = 0; a < agentCount; a++)
                {
                    accelerators[a]?.Dispose();
                    contexts[a]?.Dispose();
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Could not process: {ex.Message}");
            return -1;
        }

        return 0;
    }
}