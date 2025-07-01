namespace Sdcb.PaddleOCR;

/// <summary>
///  A record struct representing a single character recognition result from an OCR operation.
/// </summary>
public record struct RecognizedChar
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
    /// The index position of this character within the recognized text.
    /// </summary>
    public int Index { get; init; }

    /// <summary>
    ///  Initializes a new instance of the <see cref="RecognizedChar"/> record.
    /// </summary>
    /// <param name="character">The recognized character.</param>
    /// <param name="score">The confidence score of the character recognition.</param>
    /// <param name="index">The index position of this character within the recognized text.</param>
    public RecognizedChar(string character, float score, int index)
    {
        Character = character;
        Score = score;
        Index = index;
    }
}