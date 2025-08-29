﻿using OpenCvSharp;
using Sdcb.PaddleInference;
using Sdcb.PaddleOCR.Models;
using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace Sdcb.PaddleOCR;

/// <summary>
/// The PaddleOcrDetector class is responsible for detecting text regions in an image using PaddleOCR.
/// It implements IDisposable to manage memory resources.
/// </summary>
public class PaddleOcrDetector : IDisposable
{
    /// <summary>Holds an instance of the PaddlePredictor class.</summary>
    readonly PaddlePredictor _p;

    /// <summary>Gets or sets the maximum size for resizing the input image.</summary>
    public int? MaxSize { get; set; } = 960;

    /// <summary>Gets or sets the score threshold for filtering out possible text boxes.</summary>
    public float? BoxScoreThreahold { get; set; } = 0.7f;

    /// <summary>Gets or sets the threshold to binarize the text region.</summary>
    public float? BoxThreshold { get; set; } = 0.3f;

    /// <summary>Gets or sets the minimum size of the text boxes to be considered as valid.</summary>
    public int MinSize { get; set; } = 3;

    /// <summary>Gets or sets the ratio for enlarging text boxes during post-processing.</summary>
    public float UnclipRatio { get; set; } = 1.5f;

    /// <summary>
    /// Initializes a new instance of the PaddleOcrDetector class with the provided DetectionModel and PaddleConfig.
    /// </summary>
    /// <param name="model">The DetectionModel to use.</param>
    /// <param name="configure">The device and configure of the PaddleConfig, pass null to using model's DefaultDevice.</param>
    public PaddleOcrDetector(DetectionModel model, Action<PaddleConfig>? configure = null)
    {
        PaddleConfig c = model.CreateConfig();
        model.ConfigureDevice(c, configure);

        _p = c.CreatePredictor();
    }

    /// <summary>
    /// Initializes a new instance of the PaddleOcrDetector class with the provided PaddleConfig.
    /// </summary>
    /// <param name="config">The PaddleConfig to use.</param>
    public PaddleOcrDetector(PaddleConfig config) : this(config.CreatePredictor())
    {
    }

    /// <summary>
    /// Initializes a new instance of the PaddleOcrDetector class with the provided PaddlePredictor.
    /// </summary>
    /// <param name="predictor">The PaddlePredictor to use.</param>
    public PaddleOcrDetector(PaddlePredictor predictor)
    {
        _p = predictor;
    }

    /// <summary>
    /// Initializes a new instance of the PaddleOcrDetector class with model directory path.
    /// </summary>
    /// <param name="modelDir">The path of directory containing model files.</param>
    public PaddleOcrDetector(string modelDir) : this(PaddleConfig.FromModelDir(modelDir))
    {
    }

    /// <summary>
    /// Clones the current PaddleOcrDetector instance.
    /// </summary>
    /// <returns>A new PaddleOcrDetector instance with the same configuration.</returns>
    public PaddleOcrDetector Clone()
    {
        return new PaddleOcrDetector(_p.Clone());
    }

    /// <summary>
    /// Disposes the PaddlePredictor instance.
    /// </summary>
    public void Dispose()
    {
        _p.Dispose();
    }

    /// <summary>
    /// Draws detected rectangles on the input image.
    /// </summary>
    /// <param name="src">Input image.</param>
    /// <param name="rects">Array of detected rectangles.</param>
    /// <param name="color">Color of the rectangles.</param>
    /// <param name="thickness">Thickness of the rectangle lines.</param>
    /// <returns>A new image with the detected rectangles drawn on it.</returns>
    public static Mat Visualize(Mat src, RotatedRect[] rects, Scalar color, int thickness)
    {
        Mat clone = src.Clone();
        clone.DrawContours(rects.Select(x => x.Points().Select(x => (Point)x)), -1, color, thickness);
        return clone;
    }

    /// <summary>
    /// Runs the text box detection process on the input image.
    /// </summary>
    /// <param name="src">Input image.</param>
    /// <returns>An array of detected rotated rectangles representing text boxes.</returns>
    /// <exception cref="ArgumentException">Thrown when input image is empty.</exception>
    /// <exception cref="NotSupportedException">Thrown when input image channels are not 3 or 1.</exception>
    /// <exception cref="Exception">Thrown when PaddlePredictor run fails.</exception>
    public RotatedRect[] Run(Mat src)
    {
        (int height, int width, float[] data) = RunRawCore(src, out Size resizedSize);
        GCHandle dataGcHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            using Mat pred = Mat.FromPixelData(height, width, MatType.CV_32FC1, dataGcHandle.AddrOfPinnedObject());
            using Mat roi = pred[0, resizedSize.Height, 0, resizedSize.Width];
            using Mat cbuf = new();
            roi.ConvertTo(cbuf, MatType.CV_8UC1, 255);
            using Mat binary = BoxThreshold != null ?
                cbuf.Threshold((int)(BoxThreshold * 255), 255, ThresholdTypes.Binary) :
                cbuf.Clone();

            Point[][] contours = binary.FindContoursAsArray(RetrievalModes.List, ContourApproximationModes.ApproxSimple);
            Size size = src.Size();
            double scaleRate = 1.0 * src.Width / resizedSize.Width;

            RotatedRect[] rects = contours
                .Where(x => BoxScoreThreahold == null || GetScore(x, roi) > BoxScoreThreahold)
                .Select(x =>
                {
                    RotatedRect rect = Cv2.MinAreaRect(x);

                    float area = (float)Math.Abs(Cv2.ContourArea(x));
                    float perimeter = (float)Cv2.ArcLength(x, true);
                    if (perimeter < 1e-6f)
                        return null;

                    float d = UnclipRatio * area / perimeter;

                    Size2f newSize = new(
                        (rect.Size.Width + 2 * d) * (float)scaleRate,
                        (rect.Size.Height + 2 * d) * (float)scaleRate);

                    RotatedRect enlarged = new(rect.Center * (float)scaleRate, newSize, rect.Angle);

                    return (RotatedRect?)enlarged;
                })
                .Where(r => r.HasValue &&
                            r.Value.Size.Width > MinSize &&
                            r.Value.Size.Height > MinSize)
                .Select(v => v!.Value)
                .OrderBy(v => v.Center.Y)
                .ThenBy(v => v.Center.X)
                .ToArray();
            return rects;
        }
        finally
        {
            dataGcHandle.Free();
        }
    }

    /// <summary>
    /// Runs detection on the provided input image and returns the detected image as a <see cref="MatType.CV_32FC1"/> <see cref="Mat"/> object.
    /// </summary>
    /// <param name="src">The input image to run detection model on.</param>
    /// <param name="resizedSize">The returned image actuall size without padding.</param>
    /// <returns>the detected image as a <see cref="MatType.CV_32FC1"/> <see cref="Mat"/> object.</returns>
    public Mat RunRaw(Mat src, out Size resizedSize)
    {
        (int height, int width, float[] data) = RunRawCore(src, out resizedSize);
        GCHandle dataGcHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            using Mat temp = Mat.FromPixelData(height, width, MatType.CV_32FC1, data);
            return temp.Clone();
        }
        finally
        {
            dataGcHandle.Free();
        }
    }

    private (int height, int width, float[] data) RunRawCore(Mat src, out Size resizedSize)
    {
        if (src.Empty())
        {
            throw new ArgumentException("src size should not be 0, wrong input picture provided?");
        }

        if (!(src.Channels() switch { 3 or 1 => true, _ => false }))
        {
            throw new NotSupportedException($"{nameof(src)} channel must be 3 or 1, provided {src.Channels()}.");
        }
        Mat padded;
        using (Mat resized = MatResize(src, MaxSize))
        {
            resizedSize = new Size(resized.Width, resized.Height);
            padded = MatPadding32(resized);
        }

        Mat normalized;
        using (Mat _ = padded)
        {
            normalized = Normalize(padded);
        }

        PaddlePredictor predictor = _p;
        lock (predictor)
        {
            using (Mat _ = normalized)
            using (PaddleTensor input = predictor.GetInputTensor(predictor.InputNames[0]))
            {
                input.Shape = new[] { 1, 3, normalized.Rows, normalized.Cols };
                float[] data = ExtractMat(normalized);
                input.SetData(data);
            }

            if (!predictor.Run())
            {
                throw new Exception("PaddlePredictor(Detector) run failed.");
            }

            using (PaddleTensor output = predictor.GetOutputTensor(predictor.OutputNames[0]))
            {
                float[] data = output.GetData<float>();
                int[] shape = output.Shape;

                return (shape[2], shape[3], data);
            }
        }
    }

    private static float GetScore(Point[] contour, Mat pred)
    {
        int width = pred.Width;
        int height = pred.Height;

        // Clamp contour points to bounds to avoid negatives/out-of-range
        Point[] clampedPoints = contour
            .Select(p => new Point(MathUtil.Clamp(p.X, 0, width - 1), MathUtil.Clamp(p.Y, 0, height - 1)))
            .ToArray();

        int xmin = clampedPoints.Min(p => p.X);
        int xmax = clampedPoints.Max(p => p.X);
        int ymin = clampedPoints.Min(p => p.Y);
        int ymax = clampedPoints.Max(p => p.Y);

        int roiWidth = xmax - xmin + 1;
        int roiHeight = ymax - ymin + 1;
        if (roiWidth <= 0 || roiHeight <= 0)
        {
            return 0f;
        }

        Point[] rootPoints = clampedPoints
            .Select(v => new Point(v.X - xmin, v.Y - ymin))
            .ToArray();

        using Mat mask = new(roiHeight, roiWidth, MatType.CV_8UC1, Scalar.Black);
        mask.FillPoly(new[] { rootPoints }, new Scalar(255));

        if (Cv2.CountNonZero(mask) == 0)
        {
            return 0f;
        }

        using Mat croppedMat = pred[ymin, ymax + 1, xmin, xmax + 1];
        float score = (float)croppedMat.Mean(mask).Val0;
        return score;
    }

    private static Mat MatResize(Mat src, int? maxSize)
    {
        if (maxSize == null) return src.Clone();

        Size size = src.Size();
        int longEdge = Math.Max(size.Width, size.Height);
        double scaleRate = 1.0 * maxSize.Value / longEdge;

        return scaleRate < 1.0 ?
            src.Resize(default, scaleRate, scaleRate) :
            src.Clone();
    }

    internal static float[] ExtractMat(Mat src)
    {
        int rows = src.Rows;
        int cols = src.Cols;
        float[] result = new float[rows * cols * 3];
        GCHandle resultHandle = default;
        try
        {
            resultHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr resultPtr = resultHandle.AddrOfPinnedObject();
            for (int i = 0; i < src.Channels(); ++i)
            {
                using Mat dest = Mat.FromPixelData(rows, cols, MatType.CV_32FC1, resultPtr + i * rows * cols * sizeof(float));
                Cv2.ExtractChannel(src, dest, i);
            }
        }
        finally
        {
            resultHandle.Free();
        }
        return result;
    }

    private static Mat MatPadding32(Mat src)
    {
        Size size = src.Size();
        Size newSize = new(
            32 * Math.Ceiling(1.0 * size.Width / 32),
            32 * Math.Ceiling(1.0 * size.Height / 32));
        return src.CopyMakeBorder(0, newSize.Height - size.Height, 0, newSize.Width - size.Width, BorderTypes.Constant, Scalar.Black);
    }

    private static Mat Normalize(Mat src)
    {
        using Mat normalized = new();
        src.ConvertTo(normalized, MatType.CV_32FC3, 1.0 / 255);
        Mat[] bgr = normalized.Split();
        float[] scales = new[] { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
        float[] means = new[] { 0.485f, 0.456f, 0.406f };
        for (int i = 0; i < bgr.Length; ++i)
        {
            bgr[i].ConvertTo(bgr[i], MatType.CV_32FC1, 1.0 * scales[i], (0.0 - means[i]) * scales[i]);
        }

        Mat dest = new();
        Cv2.Merge(bgr, dest);

        foreach (Mat channel in bgr)
        {
            channel.Dispose();
        }

        return dest;
    }
}
