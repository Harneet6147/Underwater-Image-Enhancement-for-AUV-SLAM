#include <iostream>
#include <iomanip>
#include <filesystem>
#include "enhancer.hpp"
#include "feature_analyzer.hpp"
#include "metrics.hpp"
#include "detector.hpp"

namespace fs = std::filesystem;

void printHeader(const std::string &title)
{
    std::cout << "\n========== " << title << " ==========\n";
}

void printDetectionSummary(const std::string &label,
                           const std::vector<underwater::Detection> &detections)
{
    std::cout << "\n"
              << label << ":\n";
    std::cout << "  Total objects found: " << detections.size() << "\n";

    // Count by class
    std::map<std::string, int> classCounts;
    for (const auto &det : detections)
    {
        classCounts[det.className]++;
    }

    if (!classCounts.empty())
    {
        std::cout << "  Breakdown:\n";
        for (const auto &[name, count] : classCounts)
        {
            std::cout << "    - " << name << ": " << count << "\n";
        }
    }
}

int main(int argc, char **argv)
{
    std::cout << R"(
  ╔═══════════════════════════════════════════════════╗
  ║  Underwater Enhancement & Trash Detection System  ║
  ║        For Marine Environment Protection          ║
  ╚═══════════════════════════════════════════════════╝
    )" << "\n";

    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <image_path> [output_dir] [model_dir]\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  image_path  - Path to underwater image\n";
        std::cout << "  output_dir  - Output directory (default: output/)\n";
        std::cout << "  model_dir   - Directory containing YOLO model files (optional)\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " underwater.jpg results/ models/\n";
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputDir = (argc > 2) ? argv[2] : "output";
    std::string modelDir = (argc > 3) ? argv[3] : "models";

    fs::create_directories(outputDir);

    // ==================== LOAD IMAGE ====================
    cv::Mat original = cv::imread(inputPath);
    if (original.empty())
    {
        std::cerr << "Error: Could not load image: " << inputPath << "\n";
        return 1;
    }

    std::cout << "Input: " << inputPath << "\n";
    std::cout << "Size:  " << original.cols << "x" << original.rows << "\n";

    // ==================== ENHANCEMENT ====================
    printHeader("STAGE 1: IMAGE ENHANCEMENT");

    underwater::Enhancer enhancer;
    cv::Mat enhanced = enhancer.enhance(original);

    std::cout << "Applied:\n";
    std::cout << "  [✓] Red channel compensation\n";
    std::cout << "  [✓] Gray World white balance\n";
    std::cout << "  [✓] CLAHE contrast enhancement\n";

    // Quality metrics
    double origUIQM = underwater::ImageMetrics::computeUIQM(original);
    double enhUIQM = underwater::ImageMetrics::computeUIQM(enhanced);
    double origColor = underwater::ImageMetrics::computeColorfulness(original);
    double enhColor = underwater::ImageMetrics::computeColorfulness(enhanced);

    std::cout << "\nQuality Improvement:\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  UIQM:        " << origUIQM << " → " << enhUIQM
              << " (+" << ((enhUIQM - origUIQM) / origUIQM * 100) << "%)\n";
    std::cout << "  Colorfulness: " << origColor << " → " << enhColor
              << " (+" << ((enhColor - origColor) / origColor * 100) << "%)\n";

    // ==================== FEATURE ANALYSIS ====================
    printHeader("STAGE 2: FEATURE ANALYSIS");

    underwater::FeatureAnalyzer analyzer;
    auto origKp = analyzer.detectORB(original);
    auto enhKp = analyzer.detectORB(enhanced);

    auto origStats = analyzer.computeStats(original, origKp, "ORB");
    auto enhStats = analyzer.computeStats(enhanced, enhKp, "ORB");

    std::cout << "ORB Features:\n";
    std::cout << "  Original: " << origStats.numKeypoints << " keypoints, "
              << std::setprecision(0) << (origStats.coverage * 100) << "% coverage\n";
    std::cout << "  Enhanced: " << enhStats.numKeypoints << " keypoints, "
              << (enhStats.coverage * 100) << "% coverage\n";

    double featureImprovement = (enhStats.numKeypoints - origStats.numKeypoints) * 100.0 /
                                std::max(origStats.numKeypoints, 1);
    std::cout << "  Improvement: " << std::showpos << std::setprecision(1)
              << featureImprovement << "% more features\n";
    std::cout << std::noshowpos;

    // ==================== TRASH DETECTION ====================
    printHeader("STAGE 3: TRASH DETECTION");

    underwater::TrashDetector detector;
    std::string modelPath = modelDir + "/yolov4-tiny.weights";
    std::string configPath = modelDir + "/yolov4-tiny.cfg";
    std::string classesPath = modelDir + "/trash_classes.txt";

    bool detectionEnabled = false;
    std::vector<underwater::Detection> origDetections, enhDetections;

    if (fs::exists(modelPath) && fs::exists(configPath) && fs::exists(classesPath))
    {
        if (detector.loadModel(modelPath, configPath, classesPath))
        {
            detectionEnabled = true;

            origDetections = detector.detect(original);
            enhDetections = detector.detect(enhanced);

            printDetectionSummary("Original Image", origDetections);
            printDetectionSummary("Enhanced Image", enhDetections);

            if (enhDetections.size() > origDetections.size())
            {
                std::cout << "\n★ Enhancement helped detect "
                          << (enhDetections.size() - origDetections.size())
                          << " more objects!\n";
            }
        }
    }
    else
    {
        std::cout << "Model files not found in: " << modelDir << "\n";
        std::cout << "Skipping detection. To enable, add:\n";
        std::cout << "  - yolov4-tiny.weights\n";
        std::cout << "  - yolov4-tiny.cfg\n";
        std::cout << "  - trash_classes.txt\n";
        std::cout << "\nSee README.md for download instructions.\n";
    }

    // ==================== SAVE RESULTS ====================
    printHeader("SAVING RESULTS");

    // Feature visualizations
    cv::Mat origFeatureVis = analyzer.visualizeKeypoints(original, origKp);
    cv::Mat enhFeatureVis = analyzer.visualizeKeypoints(enhanced, enhKp);

    // Save images
    cv::imwrite(outputDir + "/1_original.jpg", original);
    cv::imwrite(outputDir + "/2_enhanced.jpg", enhanced);
    cv::imwrite(outputDir + "/3_original_features.jpg", origFeatureVis);
    cv::imwrite(outputDir + "/4_enhanced_features.jpg", enhFeatureVis);

    // Enhancement steps
    cv::imwrite(outputDir + "/step_a_color_corrected.jpg", enhancer.getColorCorrected());
    cv::imwrite(outputDir + "/step_b_white_balanced.jpg", enhancer.getWhiteBalanced());

    // Detection results (if enabled)
    if (detectionEnabled)
    {
        cv::Mat origDetVis = detector.visualize(original, origDetections);
        cv::Mat enhDetVis = detector.visualize(enhanced, enhDetections);
        cv::imwrite(outputDir + "/5_original_detections.jpg", origDetVis);
        cv::imwrite(outputDir + "/6_enhanced_detections.jpg", enhDetVis);

        // Side-by-side comparison
        cv::Mat comparison;
        cv::hconcat(origDetVis, enhDetVis, comparison);
        cv::putText(comparison, "Original", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        cv::putText(comparison, "Enhanced", cv::Point(original.cols + 10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        cv::imwrite(outputDir + "/7_detection_comparison.jpg", comparison);
    }

    std::cout << "Results saved to: " << outputDir << "/\n";

    // ==================== SUMMARY ====================
    printHeader("SUMMARY");
    std::cout << R"(
This pipeline demonstrates:

1. ENHANCEMENT  - Underwater images are degraded by light absorption
                  and scattering. We restore color and contrast.

2. FEATURES     - Better images = more features for robot navigation
                  (Visual SLAM needs features to work!)

3. DETECTION    - Clearer images help detect marine debris
                  (Robots can find and map ocean trash!)

★ Application: Bio-inspired underwater robots that can:
  - Navigate safely without disturbing marine life
  - Detect and locate plastic pollution
  - Monitor coral reef health
  - Help protect our oceans! 🌊
)";

    std::cout << "\n✓ Analysis complete!\n";
    return 0;
}