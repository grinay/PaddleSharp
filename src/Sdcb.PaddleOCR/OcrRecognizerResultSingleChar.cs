namespace Sdcb.PaddleOCR;

/// <summary>
///  A record struct representing a single character recognition result from an OCR operation.
/// </summary>
public record struct OcrRecognizerResultSingleChar
{
    /// <summary>
    /// A single character recognized from the image.
    /// </summary>
    public string Character { get; init; }

    /// <summary>
    /// The confidence score of the text recognition.
    /// </summary>
    public float Score { get; init; }

    /// <summary>
    ///  Initializes a new instance of the <see cref="OcrRecognizerResultSingleChar"/> record.
    /// </summary>
    public OcrRecognizerResultSingleChar(string character, float score)
    {
        Character = character;
        Score = score;
    }
}