#ifndef ENHANCER_HPP
#define ENHANCER_HPP

#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;

namespace underwater
{

    class Enhancer
    {
    public:
        Enhancer() = default;
        ~Enhancer() = default;

        // White balance using Gray World assumption
        Mat applyWhiteBalance(const Mat &input);

        // CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Mat applyCLAHE(const cv::Mat &input, double clipLimit = 2.0,
                           cv::Size tileGridSize = Size(8, 8));

        // Color correction for underwater (compensate red channel loss)
        Mat applyColorCorrection(const Mat &input);

        // Combined enhancement pipeline
        Mat enhance(const Mat &input);

        // Getters for intermediate results
        Mat getWhiteBalanced() const { return whiteBalanced_; }
        Mat getColorCorrected() const { return colorCorrected_; }
        Mat getCLAHEResult() const { return claheResult_; }

    private:
        Mat whiteBalanced_;
        Mat colorCorrected_;
        Mat claheResult_;
    };

} // namespace underwater

#endif // ENHANCER_HPP