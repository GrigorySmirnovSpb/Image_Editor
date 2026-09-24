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

public class myImage
{
    public int Width { get; set; }
    public int Height { get; set; }
    public byte[] Data { get; set; }
    public string Name { get; set; }
    public myImage(int  width, int height, byte [] data, string name)
    {
        Width = width;
        Height = height;
        Data = data;
        Name = name;
    }
}

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
    public static myImage LoadAsImage (string filename)
    {
        var img = Image.Load<L8> (filename);
        byte[] buf = new byte[img.Width * img.Height];
        int index  = 0;
        for (int i = 0; i < img.Height; i++)
        {
            for (int j = 0; j < img.Width; j++)
            {
                buf[index++] = img[j,i].PackedValue;
            }
        }
        return new myImage(img.Width, img.Height, buf, filename);
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
    public static void SaveImage (myImage image, string filename)
    {
        var img = Image.LoadPixelData<L8>(image.Data, image.Width, image.Height);
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
    public static byte[,] applyFilterDataParallel(float[,] filter, byte[,] image, int processorCount)
    {
        var imgH = image.GetLength(0);
        var imgW = image.GetLength(1);
        var filterD = filter.GetLength(0) / 2;

        byte[,] result = new byte[imgH, imgW];

        var options = new ParallelOptions { MaxDegreeOfParallelism = processorCount};
        Parallel.For(0, imgH, options, y =>
        {
            for (int x = 0; x < imgW; x++)
            {
                result[y, x] = ProcessPixel(filter, image, y, x, imgH, imgW, filterD);
            }
        });

        return result;
    }

    private static byte ProcessPixel(float[,] filter, byte[,] image, int px, int py, int imgH, int imgW, int filterD)
    {
        float sum = 0;
        for (int i = px - filterD; i <= px + filterD; i++)
        {
            for (int j = py - filterD; j <= py + filterD; j++)
            {
                if (i < 0 || i >= imgH || j < 0 || j >= imgW)
                {
                    sum += image[px, py] * filter[i - px + filterD, j - py + filterD];
                }
                else
                {
                    sum += image[i, j] * filter[i - px + filterD, j - py + filterD];
                }
            }
        }
        return (byte)Math.Clamp(sum, 0, 255);
    }

    public static int ProcessByData(string[] files, List<float[,]> filters, int processorCount)
    {
        int n = files.Length;
        byte[][,] images = new byte[n][,];
        byte[][,] results = new byte[n][,];

        try
        {
            Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = processorCount }, i =>
            {
                images[i] = IOImage.LoadAs2DArray(files[i]);
            });

            Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = processorCount }, i =>
            {
                int filterIndex = Math.Min(i, filters.Count - 1);
                results[i] = ConvolutionFunctions.applyFilterDataParallel(filters[filterIndex], images[i], processorCount);
            });

            Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = processorCount }, i =>
            {
                string outName = Path.Combine(
                    Path.GetDirectoryName(files[i]) ?? ".",
                    Path.GetFileNameWithoutExtension(files[i]) + "_out.png");
                IOImage.Save2DbyteArrayAsImage(results[i], outName);
                Console.WriteLine($"Saved: {outName}");
            });
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Could not process: {ex.Message}");
            return -1;
        }

        return 0;
    }

    public static int ProcessByTasks(string[] files, List<float[,]> filters, int agentCount)
    {
        int n = files.Length;
        int readerCount = Math.Max(1, agentCount / 2);
        int writerCount = Math.Max(1, agentCount / 2);

        var pendingFiles = new ConcurrentQueue<int>(Enumerable.Range(0, n));
        var toProcess = new BlockingCollection<(byte[,] Image, float[,] Filter, string OutName)>(agentCount * 2);
        var toWrite = new BlockingCollection<(byte[,] Result, string OutName)>(agentCount * 2);

        try
        {
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

            var workers = new Task[agentCount];
            for (int w = 0; w < agentCount; w++)
            {
                int local_w = w;
                workers[local_w ] = Task.Run(() =>
                {
                    foreach ((byte[,] Image, float[,] Filter, string OutName) item in toProcess.GetConsumingEnumerable())
                    {
                        byte[,] result = ConvolutionFunctions.applyFilterDataParallel(item.Filter, item.Image, agentCount);
                        toWrite.Add((result, item.OutName));
                    }
                });
            }

            var writers = new Task[writerCount];
            for (int w = 0; w < writerCount; w++)
            {
                int local_w = w;
                writers[local_w ] = Task.Run(() =>
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
        catch (Exception ex)
        {
            Console.WriteLine($"Could not process: {ex.Message}");
            return -1;
        }

        return 0;
    }
}