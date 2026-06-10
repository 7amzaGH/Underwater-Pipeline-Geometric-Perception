#include "pipeline_geometry.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <mask_points_csv> [image_width] [image_height] [direction_threshold_deg]\n\n"
                  << "Example:\n"
                  << "  " << argv[0] << " ../examples/sample_mask_points.csv\n"
                  << "  " << argv[0] << " ../examples/sample_mask_points.csv 640 640 5.0\n";

        return EXIT_FAILURE;
    }

    const std::string csv_path = argv[1];
    int image_width = 640;
    int image_height = 640;
    double direction_threshold_deg = 5.0;

    if (argc >= 3)
    {
        image_width = std::stoi(argv[2]);
    }

    if (argc >= 4)
    {
        image_height = std::stoi(argv[3]);
    }

    if (argc >= 5)
    {
        direction_threshold_deg = std::stod(argv[4]);
    }

    try
    {
        const auto points = pipeline::load_mask_points_csv(csv_path);

        const pipeline::PipelineGeometryEstimator estimator(
            image_width,
            image_height,
            50,
            direction_threshold_deg
        );

        const auto result = estimator.estimate(points);

        std::cout << "Input file: " << csv_path << "\n";
        std::cout << "Image size: " << image_width << "x" << image_height << "\n";
        std::cout << "Direction threshold: " << direction_threshold_deg << " deg\n\n";

        pipeline::print_geometry_result(result);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Runtime error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
