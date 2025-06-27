using OpenCvSharp;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR.Models.Local;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace Sdcb.PaddleOCR.Tests;

public class SingleCharTest(ITestOutputHelper testOutputHelper)
{
    [Fact]
    public void SingleCharRecognitionTest()
    {
        testOutputHelper.WriteLine(
            $"Running SingleChar test on {RuntimeInformation.OSDescription} ({RuntimeInformation.OSArchitecture})");

        // Skip for known problematic platforms
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX) && RuntimeInformation.OSArchitecture == Architecture.Arm64)
        {
            testOutputHelper.WriteLine(
                "Skipping SingleChar test on macOS arm64 due to known issues: https://github.com/PaddlePaddle/Paddle/issues/72413");
            return;
        }

        FullOcrModel model = LocalFullModels.ChineseV5;
        byte[] sampleImageData = File.ReadAllBytes(@"./samples/vsext.png");

        using PaddleOcrAll all = new(model)
        {
            AllowRotateDetection = true,
            Enable180Classification = false,
        };

        using Mat src = Cv2.ImDecode(sampleImageData, ImreadModes.Color);
        PaddleOcrResult result = all.Run(src);
        testOutputHelper.WriteLine("Detected all texts: \n" + result.Text);

        Assert.NotEmpty(result.Regions);

        foreach (PaddleOcrResultRegion region in result.Regions)
        {
            testOutputHelper.WriteLine($"Text: {region.Text}, Score: {region.Score}");
            testOutputHelper.WriteLine($"SingleChars count: {region.SingleChars.Count}");

            // Verify single characters exist
            Assert.NotEmpty(region.SingleChars);

            // Verify indices are properly set and sequential
            for (int i = 0; i < region.SingleChars.Count; i++)
            {
                OcrRecognizerResultSingleChar singleChar = region.SingleChars[i];
                Assert.Equal(i, singleChar.Index);
                Assert.NotNull(singleChar.Character);
                Assert.True(singleChar.Score > 0, $"Character '{singleChar.Character}' should have a positive score");

                testOutputHelper.WriteLine(
                    $"  Char[{singleChar.Index}]: '{singleChar.Character}', Score: {singleChar.Score:F3}");
            }

            // Verify the concatenated single characters match the full text
            string reconstructedText = string.Join("", region.SingleChars.Select(c => c.Character));
            Assert.Equal(region.Text, reconstructedText);
        }
    }
}