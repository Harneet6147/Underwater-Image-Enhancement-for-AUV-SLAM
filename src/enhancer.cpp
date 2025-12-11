#include "enhancer.hpp"

namespace underwater
{

    cv::Mat Enhancer::applyWhiteBalance(const cv::Mat &input)
    {
        // Gray World assumption: average of each channel should be gray
        cv::Mat result;
        input.copyTo(result);

        std::vector<cv::Mat> channels;
        cv::split(result, channels);

        double avgB = cv::mean(channels[0])[0];
        double avgG = cv::mean(channels[1])[0];
        double avgR = cv::mean(channels[2])[0];
        double avgGray = (avgB + avgG + avgR) / 3.0;

        // Scale each channel
        channels[0] = channels[0] * (avgGray / avgB);
        channels[1] = channels[1] * (avgGray / avgG);
        channels[2] = channels[2] * (avgGray / avgR);

        cv::merge(channels, result);
        result.convertTo(result, CV_8UC3);

        return result;
    }

    cv::Mat Enhancer::applyCLAHE(const cv::Mat &input, double clipLimit,
                                 cv::Size tileGridSize)
    {
        cv::Mat labImage, result;
        cv::cvtColor(input, labImage, cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> labChannels;
        cv::split(labImage, labChannels);

        // Apply CLAHE to L channel only
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);
        clahe->apply(labChannels[0], labChannels[0]);

        cv::merge(labChannels, labImage);
        cv::cvtColor(labImage, result, cv::COLOR_Lab2BGR);

        return result;
    }

    cv::Mat Enhancer::applyColorCorrection(const cv::Mat &input)
    {
        // Compensate for red channel attenuation underwater
        cv::Mat result;
        input.copyTo(result);
        result.convertTo(result, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels;
        cv::split(result, channels);

        // Calculate mean intensities
        double meanR = cv::mean(channels[2])[0];
        double meanG = cv::mean(channels[1])[0];
        double meanB = cv::mean(channels[0])[0];

        // Boost red channel based on green (red attenuates faster)
        if (meanR > 0)
        {
            double redGain = std::min(meanG / meanR, 1.5);
            channels[2] = channels[2] * redGain;
        }

        cv::merge(channels, result);
        result.convertTo(result, CV_8UC3, 255.0);

        return result;
    }

    cv::Mat Enhancer::enhance(const cv::Mat &input)
    {
        // Full pipeline: Color Correction -> White Balance -> CLAHE
        colorCorrected_ = applyColorCorrection(input);
        whiteBalanced_ = applyWhiteBalance(colorCorrected_);
        claheResult_ = applyCLAHE(whiteBalanced_);

        return claheResult_;
    }

} // namespace underwater