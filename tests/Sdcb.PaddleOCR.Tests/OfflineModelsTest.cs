using OpenCvSharp;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR.Models.Local;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace Sdcb.PaddleOCR.Tests;

public class OfflineModelsTest(ITestOutputHelper console)
{
    [Fact]
    public void FastCheckOCREnglishV3()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX) && RuntimeInformation.OSArchitecture == Architecture.X64)
        {
            // macOS is onnx only, bug buggy for EnglishV3, so skip this test
            console.WriteLine("Skipping EnglishV3 test on macOS x64 due to known issues with ONNX model.");
            return;
        }

        FullOcrModel model = LocalFullModels.EnglishV3;
        FastCheck(model);
    }

    [Fact]
    public void FastCheckOCREnglishV4()
    {
        FullOcrModel model = LocalFullModels.EnglishV4;
        FastCheck(model);
    }

    [Fact]
    public void FastCheckOCRChineseV4()
    {
        FullOcrModel model = LocalFullModels.ChineseV4;
        FastCheck(model);
    }

    [Fact]
    public void FastCheckOCRChineseV5()
    {
        FullOcrModel model = LocalFullModels.ChineseV5;
        FastCheck(model);
    }

    private void FastCheck(FullOcrModel model)
    {
        // from: https://visualstudio.microsoft.com/wp-content/uploads/2021/11/Home-page-extension-visual-updated.png
        byte[] sampleImageData = File.ReadAllBytes(@"./samples/vsext.png");

        using (PaddleOcrAll all = new(model)
        {
            AllowRotateDetection = true,
            Enable180Classification = false,
        })
        {
            // Load local file by following code:
            // using (Mat src2 = Cv2.ImRead(@"C:\test.jpg"))
            using (Mat src = Cv2.ImDecode(sampleImageData, ImreadModes.Color))
            {
                PaddleOcrResult result = all.Run(src);
                console.WriteLine("Detected all texts: \n" + result.Text);
                foreach (PaddleOcrResultRegion region in result.Regions)
                {
                    console.WriteLine($"Text: {region.Text}, Score: {region.Score}, RectCenter: {region.Rect.Center}, RectSize:    {region.Rect.Size}, Angle: {region.Rect.Angle}");
                }
            }
        }
    }

    [Fact]
    public async Task QueuedOCR()
    {
        FullOcrModel model = LocalFullModels.EnglishV4;

        // from: https://visualstudio.microsoft.com/wp-content/uploads/2021/11/Home-page-extension-visual-updated.png
        byte[] sampleImageData = File.ReadAllBytes(@"./samples/vsext.png");

        using QueuedPaddleOcrAll all = new(() => new PaddleOcrAll(model)
        {
            AllowRotateDetection = true,
            Enable180Classification = false,
        }, consumerCount: 1, boundedCapacity: 64);

        {
            // Load local file by following code:
            // using (Mat src2 = Cv2.ImRead(@"C:\test.jpg"))
            using (Mat src = Cv2.ImDecode(sampleImageData, ImreadModes.Color))
            {
                PaddleOcrResult result = await all.Run(src);
                console.WriteLine("Detected all texts: \n" + result.Text);
                foreach (PaddleOcrResultRegion region in result.Regions)
                {
                    console.WriteLine($"Text: {region.Text}, Score: {region.Score}, RectCenter: {region.Rect.Center}, RectSize:    {region.Rect.Size}, Angle: {region.Rect.Angle}");
                }
            }
        }
    }
}