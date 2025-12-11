#include "feature_analyzer.hpp"

namespace underwater
{

    FeatureAnalyzer::FeatureAnalyzer()
    {
        orbDetector_ = cv::ORB::create(1000);
        siftDetector_ = cv::SIFT::create(1000);
    }

    std::vector<cv::KeyPoint> FeatureAnalyzer::detectORB(const cv::Mat &image, int nFeatures)
    {
        orbDetector_->setMaxFeatures(nFeatures);

        cv::Mat gray;
        if (image.channels() == 3)
        {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = image;
        }

        std::vector<cv::KeyPoint> keypoints;
        orbDetector_->detect(gray, keypoints);

        return keypoints;
    }

    std::vector<cv::KeyPoint> FeatureAnalyzer::detectSIFT(const cv::Mat &image, int nFeatures)
    {
        cv::Mat gray;
        if (image.channels() == 3)
        {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = image;
        }

        std::vector<cv::KeyPoint> keypoints;
        siftDetector_->detect(gray, keypoints);

        // Limit to top N by response
        if (keypoints.size() > static_cast<size_t>(nFeatures))
        {
            std::sort(keypoints.begin(), keypoints.end(),
                      [](const cv::KeyPoint &a, const cv::KeyPoint &b)
                      {
                          return a.response > b.response;
                      });
            keypoints.resize(nFeatures);
        }

        return keypoints;
    }

    FeatureStats FeatureAnalyzer::computeStats(const cv::Mat &image,
                                               const std::vector<cv::KeyPoint> &keypoints,
                                               const std::string &detectorName)
    {
        FeatureStats stats;
        stats.detectorName = detectorName;
        stats.numKeypoints = static_cast<int>(keypoints.size());

        if (keypoints.empty())
        {
            stats.avgResponse = 0.0;
            stats.avgSize = 0.0;
            stats.coverage = 0.0;
            return stats;
        }

        double totalResponse = 0.0;
        double totalSize = 0.0;

        for (const auto &kp : keypoints)
        {
            totalResponse += std::abs(kp.response);
            totalSize += kp.size;
        }

        stats.avgResponse = totalResponse / keypoints.size();
        stats.avgSize = totalSize / keypoints.size();
        stats.coverage = computeCoverage(image, keypoints);

        return stats;
    }

    double FeatureAnalyzer::computeCoverage(const cv::Mat &image,
                                            const std::vector<cv::KeyPoint> &keypoints)
    {
        // Divide image into grid and check feature distribution
        const int gridSize = 8;
        cv::Mat grid = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);

        float cellW = static_cast<float>(image.cols) / gridSize;
        float cellH = static_cast<float>(image.rows) / gridSize;

        for (const auto &kp : keypoints)
        {
            int gx = std::min(static_cast<int>(kp.pt.x / cellW), gridSize - 1);
            int gy = std::min(static_cast<int>(kp.pt.y / cellH), gridSize - 1);
            grid.at<uchar>(gy, gx) = 1;
        }

        int filledCells = cv::countNonZero(grid);
        return static_cast<double>(filledCells) / (gridSize * gridSize);
    }

    cv::Mat FeatureAnalyzer::visualizeKeypoints(const cv::Mat &image,
                                                const std::vector<cv::KeyPoint> &keypoints)
    {
        cv::Mat output;
        cv::drawKeypoints(image, keypoints, output, cv::Scalar(0, 255, 0),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        return output;
    }

    void FeatureAnalyzer::compareFeatures(const cv::Mat &original, const cv::Mat &enhanced,
                                          FeatureStats &origStats, FeatureStats &enhStats)
    {
        auto origKp = detectORB(original);
        auto enhKp = detectORB(enhanced);

        origStats = computeStats(original, origKp, "ORB");
        enhStats = computeStats(enhanced, enhKp, "ORB");
    }

} // namespace underwater