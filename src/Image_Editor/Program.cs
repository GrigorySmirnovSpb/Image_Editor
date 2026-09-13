using Image_Editor_lib;
using System;

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

        for (int i = 0; i < files.Length; i++)
        {
            string filename = files[i];
            int filterIndex = Math.Min(i, filters.Count - 1);
            try
            {
                byte[,] image = IOImage.LoadAs2DArray(filename);
                byte[,] res = ConvolutionFunctions.applyFilter(filters[filterIndex], image);

                string outName = Path.Combine(
                    Path.GetDirectoryName(filename) ?? ".",
                    Path.GetFileNameWithoutExtension(filename) + "_out.png");
                IOImage.Save2DbyteArrayAsImage(res, outName);
                Console.WriteLine($"Saved: {outName}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Could not apply filter to {filename}: {ex.Message}");
                return -1;
            }
        }

        return 0;
    }
}