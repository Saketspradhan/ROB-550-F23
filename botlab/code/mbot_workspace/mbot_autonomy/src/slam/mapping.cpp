#include <slam/mapping.hpp>
#include <utils/grid_utils.hpp>
#include <numeric>
#include <chrono>
using namespace std::chrono;

Mapping::Mapping(float maxLaserDistance, int8_t hitOdds, int8_t missOdds)
: kMaxLaserDistance_(maxLaserDistance)
, kHitOdds_(hitOdds)
, kMissOdds_(missOdds)
, initialized_(false)
{
}


void Mapping::updateMap(const mbot_lcm_msgs::lidar_t& scan,
                        const mbot_lcm_msgs::pose2D_t& pose,
                        OccupancyGrid& map)
{
    if (!initialized_)
        previousPose_ = pose;
    initialized_ = true;

    MovingLaserScan movingScan(scan, previousPose_, pose);

    for (auto& ray : movingScan)
    {
        scoreEndpoint(ray, map);
        scoreRay(ray, map);
    }

    previousPose_ = pose;
}

void Mapping::scoreEndpoint(const adjusted_ray_t& ray, OccupancyGrid& map)
{
    /// TODO: Implement how to score the cell that the laser endpoint hits  
    if (ray.range < kMaxLaserDistance_)
    {
        Point<float> rayStart = global_position_to_grid_cell(ray.origin, map);
        Point<int> rayCell;

        rayCell.x = static_cast<int>(rayStart.x + (ray.range * std::cos(ray.theta) * map.cellsPerMeter()));
        rayCell.y = static_cast<int>(rayStart.y + (ray.range * std::sin(ray.theta) * map.cellsPerMeter()));

        if(map.isCellInGrid(rayCell.x, rayCell.y))
        {
            increaseCellOdds(rayCell.x, rayCell.y, map);
        }
    }
    
}


void Mapping::scoreRay(const adjusted_ray_t& ray, OccupancyGrid& map)
{
    std::vector<Point<int>> cellsAlongRay = bresenham(ray, map);

    for (const Point<int>& cell : cellsAlongRay)
    {
        decreaseCellOdds(cell.x, cell.y, map);
    }
}

void Mapping::increaseCellOdds(int x, int y, OccupancyGrid& map)
{
    if(!initialized_)
    {
        // return;
    }
    else if(100 - map(x, y) > kHitOdds_)
    {
        map(x, y) += kHitOdds_;
    }
    else
    {
        map(x, y) = 100;
    }
}

void Mapping::decreaseCellOdds(int x, int y, OccupancyGrid& map)
{
    if(!initialized_)
    {
        // return;
    }
    else if(map(x, y) + 100 > kMissOdds_)
    {
        map(x, y) -= kMissOdds_;
    }
    else
    {
        map(x, y) = -100;
    }
}

/*
Takes the ray and map, and returns a vector of map cells to check
*/
std::vector<Point<int>> Mapping::bresenham(const adjusted_ray_t& ray, const OccupancyGrid& map)
{
    // Convert ray origin and endpoint to grid cells (like in scoreEndpoint)
    Point<float> rayStart = global_position_to_grid_cell(ray.origin, map);
    Point<float> rayEnd;
    rayEnd.x = rayStart.x + ray.range * std::cos(ray.theta) * map.cellsPerMeter();
    rayEnd.y = rayStart.y + ray.range * std::sin(ray.theta) * map.cellsPerMeter();

    int x0 = static_cast<int>(rayStart.x);
    int y0 = static_cast<int>(rayStart.y);
    int x1 = static_cast<int>(rayEnd.x);
    int y1 = static_cast<int>(rayEnd.y);

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    std::vector<Point<int>> points;

    while (true) {
        if (map.isCellInGrid(x0, y0)) {
            points.push_back({x0, y0});
        }

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;

        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }

    return points;
}


// std::vector<Point<int>> Mapping::divideAndStepAlongRay(const adjusted_ray_t& ray, const OccupancyGrid& map)
// {
//     /// TODO: Implement an alternative approach to find cells touched by the ray. 
//     return {};
    
// }

