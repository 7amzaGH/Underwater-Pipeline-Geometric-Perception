#ifndef PIPELINE_GEOMETRY_HPP
#define PIPELINE_GEOMETRY_HPP

#include <string>
#include <vector>

namespace pipeline
{

struct Point2D
{
    double x = 0.0;
    double y = 0.0;
};

enum class DirectionCue
{
    LEFT,
    STRAIGHT,
    RIGHT,
    UNKNOWN
};

struct GeometryResult
{
    bool valid = false;
    std::size_t num_points = 0;

    double center_x = 0.0;
    double center_y = 0.0;
    double center_offset_px = 0.0;

    double orientation_deg = 0.0;
    DirectionCue direction = DirectionCue::UNKNOWN;
};

class PipelineGeometryEstimator
{
public:
    PipelineGeometryEstimator(
        int image_width = 640,
        int image_height = 640,
        std::size_t min_points = 50,
        double direction_threshold_deg = 5.0
    );

    GeometryResult estimate(const std::vector<Point2D>& points) const;

    int image_width() const;
    int image_height() const;
    std::size_t min_points() const;
    double direction_threshold_deg() const;

private:
    int image_width_;
    int image_height_;
    std::size_t min_points_;
    double direction_threshold_deg_;
};

std::vector<Point2D> load_mask_points_csv(const std::string& csv_path);

std::string to_string(DirectionCue direction);

void print_geometry_result(const GeometryResult& result);

}  // namespace pipeline

#endif  // PIPELINE_GEOMETRY_HPP
