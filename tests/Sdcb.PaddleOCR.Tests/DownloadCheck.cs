using Sdcb.PaddleOCR.Models.Online;
using Xunit;
using Xunit.Abstractions;

namespace Sdcb.PaddleOCR.Tests;

#pragma warning disable CS9113 // 参数未读。
public class DownloadCheckTest(ITestOutputHelper console)
#pragma warning restore CS9113 // 参数未读。
{
    [Fact]
    public async Task DownloadCheck()
    {
        await LocalDictOnlineRecognizationModel.ChineseMobileV2.DownloadAsync();
    }
}