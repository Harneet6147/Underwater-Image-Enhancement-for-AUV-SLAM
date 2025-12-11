#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
using namespace std;
using namespace cv;

namespace underwater
{

    struct Detection
    {
        int classId;
        string className;
        float confidence;
        Rect boundingBox;
    };

    class TrashDetector
    {
    public:
        TrashDetector();
        ~TrashDetector() = default;

        // Load pre-trained model (YOLO format)
        bool loadModel(const string &modelPath,
                       const string &configPath,
                       const string &classesPath);

        // Detect trash in image
        std::vector<Detection> detect(const Mat &image,
                                      float confThreshold = 0.5,
                                      float nmsThreshold = 0.4);

        // Visualize detections on image
        cv::Mat visualize(const Mat &image,
                          const vector<Detection> &detections);

        // Compare detection performance (before vs after enhancement)
        void compareDetections(const Mat &original,
                               const Mat &enhanced,
                               vector<Detection> &origDetections,
                               vector<Detection> &enhDetections);

        // Get class names
        vector<std::string> getClassNames() const { return classNames_; }

        // Check if model is loaded
        bool isLoaded() const { return modelLoaded_; }

    private:
        dnn::Net net_;
        vector<std::string> classNames_;
        bool modelLoaded_;

        // YOLO specific
        vector<string> getOutputLayerNames();
        vector<Scalar> generateColors(int numClasses);
        vector<Scalar> colors_;
    };

} // namespace underwater

#endif // DETECTOR_HPP