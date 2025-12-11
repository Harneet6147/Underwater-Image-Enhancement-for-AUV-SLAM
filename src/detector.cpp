#include "detector.hpp"
#include <fstream>
#include <random>

namespace underwater
{

    TrashDetector::TrashDetector() : modelLoaded_(false) {}

    bool TrashDetector::loadModel(const std::string &modelPath,
                                  const std::string &configPath,
                                  const std::string &classesPath)
    {
        try
        {
            // Load class names
            std::ifstream classFile(classesPath);
            if (!classFile.is_open())
            {
                std::cerr << "Error: Could not open classes file: " << classesPath << "\n";
                return false;
            }

            std::string line;
            while (std::getline(classFile, line))
            {
                if (!line.empty())
                {
                    classNames_.push_back(line);
                }
            }

            // Load network
            net_ = cv::dnn::readNetFromDarknet(configPath, modelPath);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

            // Generate colors for each class
            colors_ = generateColors(classNames_.size());

            modelLoaded_ = true;
            std::cout << "Model loaded successfully! Classes: " << classNames_.size() << "\n";
            return true;
        }
        catch (const cv::Exception &e)
        {
            std::cerr << "Error loading model: " << e.what() << "\n";
            return false;
        }
    }

    std::vector<cv::Scalar> TrashDetector::generateColors(int numClasses)
    {
        std::vector<cv::Scalar> colors;
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> dist(100, 255);

        for (int i = 0; i < numClasses; i++)
        {
            colors.push_back(cv::Scalar(dist(rng), dist(rng), dist(rng)));
        }
        return colors;
    }

    std::vector<std::string> TrashDetector::getOutputLayerNames()
    {
        std::vector<std::string> names;
        std::vector<int> outLayers = net_.getUnconnectedOutLayers();
        std::vector<std::string> layerNames = net_.getLayerNames();

        for (int idx : outLayers)
        {
            names.push_back(layerNames[idx - 1]);
        }
        return names;
    }

    std::vector<Detection> TrashDetector::detect(const cv::Mat &image,
                                                 float confThreshold,
                                                 float nmsThreshold)
    {
        std::vector<Detection> results;

        if (!modelLoaded_)
        {
            std::cerr << "Error: Model not loaded!\n";
            return results;
        }

        // Prepare input blob
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1 / 255.0, cv::Size(416, 416),
                               cv::Scalar(0, 0, 0), true, false);
        net_.setInput(blob);

        // Forward pass
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, getOutputLayerNames());

        // Process detections
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (const auto &output : outputs)
        {
            for (int i = 0; i < output.rows; i++)
            {
                const float *data = output.ptr<float>(i);

                // Skip first 5 values (x, y, w, h, objectness)
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);

                if (confidence > confThreshold)
                {
                    int centerX = static_cast<int>(data[0] * image.cols);
                    int centerY = static_cast<int>(data[1] * image.rows);
                    int width = static_cast<int>(data[2] * image.cols);
                    int height = static_cast<int>(data[3] * image.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(static_cast<float>(confidence));
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        // Apply Non-Maximum Suppression
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        for (int idx : indices)
        {
            Detection det;
            det.classId = classIds[idx];
            det.className = (det.classId < classNames_.size()) ? classNames_[det.classId] : "unknown";
            det.confidence = confidences[idx];
            det.boundingBox = boxes[idx];
            results.push_back(det);
        }

        return results;
    }

    cv::Mat TrashDetector::visualize(const cv::Mat &image,
                                     const std::vector<Detection> &detections)
    {
        cv::Mat output = image.clone();

        for (const auto &det : detections)
        {
            cv::Scalar color = (det.classId < colors_.size()) ? colors_[det.classId] : cv::Scalar(0, 255, 0);

            // Draw bounding box
            cv::rectangle(output, det.boundingBox, color, 2);

            // Draw label background
            std::string label = det.className + " " +
                                std::to_string(static_cast<int>(det.confidence * 100)) + "%";
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                 0.5, 1, &baseLine);
            int top = std::max(det.boundingBox.y, labelSize.height);

            cv::rectangle(output,
                          cv::Point(det.boundingBox.x, top - labelSize.height - 5),
                          cv::Point(det.boundingBox.x + labelSize.width, top),
                          color, cv::FILLED);

            // Draw label text
            cv::putText(output, label, cv::Point(det.boundingBox.x, top - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        return output;
    }

    void TrashDetector::compareDetections(const cv::Mat &original,
                                          const cv::Mat &enhanced,
                                          std::vector<Detection> &origDetections,
                                          std::vector<Detection> &enhDetections)
    {
        origDetections = detect(original);
        enhDetections = detect(enhanced);
    }

} // namespace underwater