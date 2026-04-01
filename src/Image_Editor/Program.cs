using Image_Editor_lib;
using System;

class Program
{
    static int Main()
    {
        byte[,] image;
        byte[,] imageRes;
        float[,] filter;

        string filename = "";
        Console.WriteLine("Enter name of image");
        filename = Console.ReadLine();
        try 
        {
            image = IOImage.LoadAs2DArray(filename);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"File not found, try write full path\nExeption: {ex.Message}");
            return -1;
        }

        Console.WriteLine("Choose filter (write number)\n 1: Gaussian blur (default)\n 2: Edges kernel\n 3: ScaryKernel");
        string filterType = Console.ReadLine();
        switch (filterType)
        {
            case "1":
                {
                    filter = Kernels.gaussianBlurKernel;
                    break;
                }
            case "2":
                {
                    filter = Kernels.egdeKernel;
                    break;
                }
            case "3":
                {
                    filter = Kernels.scaryKernel;
                    break;
                }
            default:
                {
                    filter = Kernels.gaussianBlurKernel;
                    break;
                } 
        }

        try
        {
            imageRes = ConvolutionFunctions.applyFilter(filter, image);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Could not apply filter: {ex.Message})");
            return -1;
        }

        try
        {
            IOImage.Save2DbyteArrayAsImage(imageRes, "out.png");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Cannot save image: {ex.Message}");
            return -1;
        }

        return 0;
    }
}