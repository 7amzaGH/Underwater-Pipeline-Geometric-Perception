#include "pipeline_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace pipeline
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

std::string trim(const std::string& value)
{
    const std::string whitespace = " \t\n\r\f\v";
    const std::size_t start = value.find_first_not_of(whitespace);

    if (start == std::string::npos)
    {
        return "";
    }

    const std::size_t end = value.find_last_not_of(whitespace);
    return value.substr(start, end - start + 1);
}

std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> tokens;
    std::stringstream stream(line);
    std::string token;

    while (std::getline(stream, token, ','))
    {
        tokens.push_back(trim(token));
    }

    return tokens;
}

double radians_to_degrees(double radians)
{
    return radians * 180.0 / kPi;
}

}  // namespace

PipelineGeometryEstimator::PipelineGeometryEstimator(
    int image_width,
    int image_height,
    std::size_t min_points,
    double direction_threshold_deg
)
    : image_width_(image_width),
      image_height_(image_height),
      min_points_(min_points),
      direction_threshold_deg_(direction_threshold_deg)
{
    if (image_width_ <= 0 || image_height_ <= 0)
    {
        throw std::invalid_argument("Image dimensions must be positive.");
    }

    if (direction_threshold_deg_ < 0.0)
    {
        throw std::invalid_argument("Direction threshold must be non-negative.");
    }
}

GeometryResult PipelineGeometryEstimator::estimate(const std::vector<Point2D>& points) const
{
    GeometryResult result;
    result.num_points = points.size();

    if (points.size() < min_points_)
    {
        result.valid = false;
        result.direction = DirectionCue::UNKNOWN;
        return result;
    }

    double sum_x = 0.0;
    double sum_y = 0.0;

    for (const auto& point : points)
    {
        sum_x += point.x;
        sum_y += point.y;
    }

    const double n = static_cast<double>(points.size());
    const double mean_x = sum_x / n;
    const double mean_y = sum_y / n;

    double cov_xx = 0.0;
    double cov_xy = 0.0;
    double cov_yy = 0.0;

    for (const auto& point : points)
    {
        const double dx = point.x - mean_x;
        const double dy = point.y - mean_y;

        cov_xx += dx * dx;
        cov_xy += dx * dy;
        cov_yy += dy * dy;
    }

    cov_xx /= n;
    cov_xy /= n;
    cov_yy /= n;

    // Closed-form principal eigenvector for a 2x2 covariance matrix.
    // The vector is equivalent to the first PCA component of the foreground mask points.
    const double trace = cov_xx + cov_yy;
    const double determinant = cov_xx * cov_yy - cov_xy * cov_xy;
    const double discriminant = std::max(0.0, trace * trace / 4.0 - determinant);
    const double lambda1 = trace / 2.0 + std::sqrt(discriminant);

    double vx = cov_xy;
    double vy = lambda1 - cov_xx;

    const double norm = std::sqrt(vx * vx + vy * vy);

    if (norm < 1e-9)
    {
        // Fallback for near-degenerate masks.
        vx = 0.0;
        vy = 1.0;
    }
    else
    {
        vx /= norm;
        vy /= norm;
    }

    // Image convention: 0 degrees corresponds to a vertically aligned pipeline.
    double angle_deg = radians_to_degrees(std::atan2(vx, -vy));

    // Resolve PCA sign ambiguity to the compact [-90, 90] orientation range.
    if (angle_deg > 90.0)
    {
        angle_deg -= 180.0;
    }
    else if (angle_deg < -90.0)
    {
        angle_deg += 180.0;
    }

    result.valid = true;
    result.center_x = mean_x;
    result.center_y = mean_y;
    result.center_offset_px = mean_x - static_cast<double>(image_width_) / 2.0;
    result.orientation_deg = angle_deg;

    if (angle_deg > direction_threshold_deg_)
    {
        result.direction = DirectionCue::RIGHT;
    }
    else if (angle_deg < -direction_threshold_deg_)
    {
        result.direction = DirectionCue::LEFT;
    }
    else
    {
        result.direction = DirectionCue::STRAIGHT;
    }

    return result;
}

int PipelineGeometryEstimator::image_width() const
{
    return image_width_;
}

int PipelineGeometryEstimator::image_height() const
{
    return image_height_;
}

std::size_t PipelineGeometryEstimator::min_points() const
{
    return min_points_;
}

double PipelineGeometryEstimator::direction_threshold_deg() const
{
    return direction_threshold_deg_;
}

std::vector<Point2D> load_mask_points_csv(const std::string& csv_path)
{
    std::ifstream file(csv_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open CSV file: " + csv_path);
    }

    std::vector<Point2D> points;
    std::string line;

    // Skip header if present.
    if (!std::getline(file, line))
    {
        throw std::runtime_error("CSV file is empty: " + csv_path);
    }

    const auto first_tokens = split_csv_line(line);
    const bool first_line_is_header =
        first_tokens.empty() ||
        first_tokens[0] == "x" || first_tokens[0] == "X" ||
        first_tokens[0] == "pixel_x";

    if (!first_line_is_header && first_tokens.size() >= 2)
    {
        points.push_back({std::stod(first_tokens[0]), std::stod(first_tokens[1])});
    }

    while (std::getline(file, line))
    {
        if (trim(line).empty())
        {
            continue;
        }

        const auto tokens = split_csv_line(line);

        if (tokens.size() < 2)
        {
            throw std::runtime_error("Invalid mask point row: " + line);
        }

        points.push_back({std::stod(tokens[0]), std::stod(tokens[1])});
    }

    return points;
}

std::string to_string(DirectionCue direction)
{
    switch (direction)
    {
        case DirectionCue::LEFT:
            return "LEFT";
        case DirectionCue::STRAIGHT:
            return "STRAIGHT";
        case DirectionCue::RIGHT:
            return "RIGHT";
        case DirectionCue::UNKNOWN:
            return "UNKNOWN";
        default:
            return "UNKNOWN";
    }
}

void print_geometry_result(const GeometryResult& result)
{
    std::cout << "Pipeline Geometry Runtime\n";
    std::cout << "-------------------------\n";
    std::cout << "Valid mask: " << (result.valid ? "true" : "false") << "\n";
    std::cout << "Foreground points: " << result.num_points << "\n";

    if (!result.valid)
    {
        std::cout << "Reason: insufficient foreground points for stable PCA estimation\n";
        return;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Center x: " << result.center_x << " px\n";
    std::cout << "Center y: " << result.center_y << " px\n";
    std::cout << "Center offset: " << result.center_offset_px << " px\n";
    std::cout << "Orientation angle: " << result.orientation_deg << " deg\n";
    std::cout << "Direction: " << to_string(result.direction) << "\n";
}

}  // namespace pipeline
