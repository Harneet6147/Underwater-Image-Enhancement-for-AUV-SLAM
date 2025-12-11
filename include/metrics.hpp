#ifndef METRICS_HPP
#define METRICS_HPP

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

namespace underwater
{

    class ImageMetrics
    {
    public:
        // Underwater Image Quality Measure (UIQM) - simplified version
        static double computeUIQM(const Mat &image);

        // Peak Signal-to-Noise Ratio (if reference available)
        static double computePSNR(const Mat &original, const Mat &enhanced);

        // Contrast measure (RMS contrast)
        static double computeContrast(const Mat &image);

        // Colorfulness metric (Hasler and Süsstrunk)
        static double computeColorfulness(const Mat &image);

        // Entropy (information content)
        static double computeEntropy(const Mat &image);

    private:
        // Helper for UIQM components
        static double computeUIQMColorful(const Mat &image);
        static double computeUIQMSharpness(const Mat &image);
        static double computeUIQMContrast(const Mat &image);
    };

} // namespace underwater

#endif // METRICS_HPP