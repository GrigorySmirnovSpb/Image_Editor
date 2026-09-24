using Image_Editor_lib;
using System;
using System.Collections.Concurrent;

class Program
{
    static int Main()
    {
        string? filenames = "";
        Console.WriteLine("Enter names of images, splited ','");
        filenames = Console.ReadLine();
        if (string.IsNullOrEmpty(filenames))
        {
            Console.WriteLine("Filename cannot be empty");
            return -1;
        }
        filenames = filenames.Replace("'", "");
        string[] files = filenames.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (files.Length == 0)
        {
            Console.WriteLine("Filename cannot be empty");
            return -1;
        }

        Console.WriteLine("Enter filters (write numbers), splited ','\n 1: Gaussian blur (default)\n 2: Edges kernel\n 3: ScaryKernel");
        string? filterInput = Console.ReadLine();
        if (string.IsNullOrEmpty(filterInput))
        {
            Console.WriteLine("Filter cannot be empty");
            return -1;
        }

        string[] filterNums = filterInput.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (filterNums.Length == 0)
        {
            Console.WriteLine("Filter cannot be empty");
            return -1;
        }

        List<float[,]> filters = new List<float[,]>();
        foreach (string num in filterNums)
        {
            switch (num)
            {
                case "1":
                    filters.Add(Kernels.gaussianBlurKernel);
                    break;
                case "2":
                    filters.Add(Kernels.egdeKernel);
                    break;
                case "3":
                    filters.Add(Kernels.scaryKernel);
                    break;
                default:
                    filters.Add(Kernels.gaussianBlurKernel);
                    break;
            }
        }

        Console.WriteLine("Choose processing mode:\n 1: Data parallelism\n 2: Task parallelism (agents)");
        string? mode = Console.ReadLine();
        bool useTasks = mode == "2";

        Console.WriteLine("Enter process/agent count");
        int parallelismCount;
        try
        {
            parallelismCount = int.Parse(Console.ReadLine() ?? "");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"It is not a number: {ex.Message}");
            return -1;
        }
        if (parallelismCount < 1)
        {
            Console.WriteLine("Count must be positive");
            return -1;
        }

        if (useTasks)
        {
            return ConvolutionFunctions.ProcessByTasks(files, filters, parallelismCount);
        }
        else
        {
            return ConvolutionFunctions.ProcessByData(files, filters, parallelismCount);
        }
    }
}