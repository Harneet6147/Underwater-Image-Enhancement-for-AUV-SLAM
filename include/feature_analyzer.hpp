#ifndef FEATURE_ANALYZER_HPP
#define FEATURE_ANALYZER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
using namespace std;
using namespace cv;

namespace underwater
{

    struct FeatureStats
    {
        int numKeypoints;
        double avgResponse; // Average feature strength
        double avgSize;     // Average feature scale
        double coverage;    // Spatial distribution (0-1)
        string detectorName;
    };

    class FeatureAnalyzer
    {
    public:
        FeatureAnalyzer();
        ~FeatureAnalyzer() = default;

        // Detect features using ORB
        vector<KeyPoint> detectORB(const Mat &image, int nFeatures = 1000);

        // Detect features using SIFT
        vector<KeyPoint> detectSIFT(const Mat &image, int nFeatures = 1000);

        // Compute statistics for detected keypoints
        FeatureStats computeStats(const Mat &image,
                                  const vector<KeyPoint> &keypoints,
                                  const string &detectorName);

        // Visualize keypoints on image
        Mat visualizeKeypoints(const Mat &image,
                               const vector<KeyPoint> &keypoints);

        // Compare features between two images (before/after enhancement)
        void compareFeatures(const Mat &original, const Mat &enhanced,
                             FeatureStats &origStats, FeatureStats &enhStats);

    private:
        Ptr<ORB>
            orbDetector_;
        Ptr<SIFT> siftDetector_;

        double computeCoverage(const Mat &image,
                               const vector<KeyPoint> &keypoints);
    };

} // namespace underwater

#endif // FEATURE_ANALYZER_HPP