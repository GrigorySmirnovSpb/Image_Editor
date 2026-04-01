namespace Image_Editor_lib;
using System;
using System.Collections;
using System.IO;
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
                result[j,i] = img[i,j].PackedValue;
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
                buf[index++] = img[i,j].PackedValue;
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
    public static byte[,] applyFilter(float[,] filter, byte[,] image) 
    {
        var imgH = image.GetLength(0);
        var imgW = image.GetLength(1);

        var filterD = filter.GetLength(0) / 2;

        byte[,] result = new byte[imgH, imgW];

        byte processPixel(int px, int py)
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
            sum = sum;
            return (byte)sum;
        }

        for (int i = 0; i < imgH; i++)
        {
            for (int j = 0; j < imgW; j++)
            {
                result[i,j] = processPixel(i, j);
            }
        }
        return result;
    }
}