using OpenCvSharp;
using Sdcb.PaddleInference;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR.Models.Online;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace Sdcb.PaddleOCR.Tests;

public class OnlineModelsTest(ITestOutputHelper console)
{
    [Fact]
    public async Task FastCheckOCR()
    {
        FullOcrModel model = await OnlineFullModels.EnglishV4.DownloadAsync();

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
                Stopwatch sw = Stopwatch.StartNew();
                PaddleOcrResult result = all.Run(src);
                console.WriteLine($"lapsed={sw.ElapsedMilliseconds} ms");
                console.WriteLine("Detected all texts: \n" + result.Text);
                foreach (PaddleOcrResultRegion region in result.Regions)
                {
                    console.WriteLine($"Text: {region.Text}, Score: {region.Score}, RectCenter: {region.Rect.Center}, RectSize:    {region.Rect.Size}, Angle: {region.Rect.Angle}");
                }
            }
        }
    }

    [Fact]
    public async Task V4DetOnly()
    {
        DetectionModel detModel = await OnlineDetectionModel.ChineseV4.DownloadAsync();

        using (Mat src = Cv2.ImRead(@"./samples/5ghz.jpg"))
        using (PaddleOcrDetector r = new(detModel))
        {
            Stopwatch sw = Stopwatch.StartNew();
            RotatedRect[] rects = r.Run(src);
            console.WriteLine($"elapsed={sw.ElapsedMilliseconds}ms");
            console.WriteLine($"Detected {rects.Length} rects.");
        }
    }

    [Fact]
    public async Task V4RecOnly()
    {
        RecognizationModel recModel = await LocalDictOnlineRecognizationModel.ChineseV4.DownloadAsync();

        using (Mat src = Cv2.ImRead(@"./samples/5ghz.jpg"))
        using (PaddleOcrRecognizer r = new(recModel))
        {
            Stopwatch sw = Stopwatch.StartNew();
            PaddleOcrRecognizerResult result = r.Run(src);
            console.WriteLine($"elapsed={sw.ElapsedMilliseconds}ms");
            console.WriteLine(result.ToString());
            Assert.True(result.Score > 0.9);
        }
    }

    //[Fact]
    //public async Task V4FastCheckOCR()
    //{
    //    OnlineFullModels onlineModels = new OnlineFullModels(
    //        OnlineDetectionModel.ChineseV4, OnlineClassificationModel.ChineseMobileV2, LocalDictOnlineRecognizationModel.EnglishV4);
    //    FullOcrModel model = await onlineModels.DownloadAsync();

    //    // from: https://visualstudio.microsoft.com/wp-content/uploads/2021/11/Home-page-extension-visual-updated.png
    //    byte[] sampleImageData = File.ReadAllBytes(@"./samples/vsext.png");

    //    using (PaddleOcrAll all = new(model)
    //    {
    //        AllowRotateDetection = true,
    //        Enable180Classification = false,
    //    })
    //    {
    //        // Load local file by following code:
    //        // using (Mat src2 = Cv2.ImRead(@"C:\test.jpg"))
    //        using (Mat src = Cv2.ImDecode(sampleImageData, ImreadModes.Color))
    //        {
    //            PaddleOcrResult result = null!;
    //            Stopwatch sw = Stopwatch.StartNew();
    //            result = all.Run(src);
    //            sw.Stop();

    //            _console.WriteLine($"elapsed={sw.ElapsedMilliseconds}ms");
    //            _console.WriteLine("Detected all texts: \n" + result.Text);
    //        }
    //    }
    //}
}