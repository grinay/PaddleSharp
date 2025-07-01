using System.Collections.Generic;

namespace Sdcb.PaddleOCR;

/// <summary>
/// A struct representing the result of an image recognition operation using Paddle OCR.
/// </summary>
public readonly record struct PaddleOcrRecognizerResult
{
    /// <summary>
    /// The recognized text from the image.
    /// </summary>
    public string Text { get; init; }

    /// <summary>
    /// The confidence score of the text recognition.
    /// </summary>
    public float Score { get; init; }
    
    /// <summary>
    /// A read-only list of single character recognition results.
    /// </summary>
    public IReadOnlyList<OcrRecognizerResultSingleChar> OcrRecognizerResultSingleChars { get; init; }

    /// <summary>
    /// Initializes a new instance of the <see cref="PaddleOcrRecognizerResult"/> struct.
    /// </summary>
    /// <param name="text">The recognized text from the image.</param>
    /// <param name="score">The confidence score of the text recognition.</param>
    /// <param name="ocrRecognizerResultSingleChar">A list of single character recognition results.</param>
    public PaddleOcrRecognizerResult(string text, float score, IReadOnlyList<OcrRecognizerResultSingleChar> ocrRecognizerResultSingleChar)
    {
        Text = text;
        Score = score;
        OcrRecognizerResultSingleChars = ocrRecognizerResultSingleChar ?? new List<OcrRecognizerResultSingleChar>();
    }
}