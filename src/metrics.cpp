#include "metrics.hpp"
#include <cmath>
using namespace std;
using namespace cv;

namespace underwater
{

    double ImageMetrics::computeColorfulness(const cv::Mat &image)
    {
        // Hasler and Süsstrunk colorfulness metric
        Mat floatImg;
        image.convertTo(floatImg, CV_64FC3);

        vector<cv::Mat> channels;
        split(floatImg, channels);

        Mat rg = channels[2] - channels[1];                       // R - G
        Mat yb = 0.5 * (channels[2] + channels[1]) - channels[0]; // (R+G)/2 - B

        Scalar rgMean, rgStd, ybMean, ybStd;
        meanStdDev(rg, rgMean, rgStd);
        meanStdDev(yb, ybMean, ybStd);

        double stdRoot = std::sqrt(rgStd[0] * rgStd[0] + ybStd[0] * ybStd[0]);
        double meanRoot = std::sqrt(rgMean[0] * rgMean[0] + ybMean[0] * ybMean[0]);

        return stdRoot + 0.3 * meanRoot;
    }

    double ImageMetrics::computeContrast(const Mat &image)
    {
        Mat gray;
        if (image.channels() == 3)
        {
            cvtColor(image, gray, COLOR_BGR2GRAY);
        }
        else
        {
            gray = image;
        }

        Scalar mean, stddev;
        meanStdDev(gray, mean, stddev);

        return stddev[0]; // RMS contrast
    }

    double ImageMetrics::computeEntropy(const Mat &image)
    {
        Mat gray;
        if (image.channels() == 3)
        {
            cvtColor(image, gray, COLOR_BGR2GRAY);
        }
        else
        {
            gray = image;
        }

        // Compute histogram
        int histSize = 256;
        float range[] = {0, 256};
        const float *histRange = {range};
        Mat hist;
        calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

        // Normalize to probability
        hist /= gray.total();

        // Compute entropy
        double entropy = 0.0;
        for (int i = 0; i < histSize; i++)
        {
            float p = hist.at<float>(i);
            if (p > 0)
            {
                entropy -= p * std::log2(p);
            }
        }

        return entropy;
    }

    double ImageMetrics::computePSNR(const cv::Mat &original, const cv::Mat &enhanced)
    {
        return PSNR(original, enhanced);
    }

    double ImageMetrics::computeUIQMSharpness(const cv::Mat &image)
    {
        Mat gray, laplacian;
        if (image.channels() == 3)
        {
            cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = image;
        }

        Laplacian(gray, laplacian, CV_64F);
        Scalar mean, stddev;
        meanStdDev(laplacian, mean, stddev);

        return stddev[0] * stddev[0]; // Variance of Laplacian
    }

    double ImageMetrics::computeUIQM(const cv::Mat &image)
    {
        // Simplified UIQM: weighted combination of colorfulness, sharpness, contrast
        double c1 = 0.0282, c2 = 0.2953, c3 = 3.5753;

        double colorfulness = computeColorfulness(image);
        double sharpness = computeUIQMSharpness(image);
        double contrast = computeContrast(image);

        return c1 * colorfulness + c2 * sharpness + c3 * contrast;
    }

} // namespace underwater