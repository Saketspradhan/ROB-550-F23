#include <slam/sensor_model.hpp>
#include <slam/moving_laser_scan.hpp>
#include <slam/occupancy_grid.hpp>
#include <mbot_lcm_msgs/particle_t.hpp>
#include <utils/grid_utils.hpp>
#include <utils/geometric/point.hpp>
SensorModel::SensorModel(void)
:   sigma_hit_(0.075),
	occupancy_threshold_(10),
	ray_stride_(7),
	max_ray_range_(1000),
    search_range(2),
    offset_quality_weight(3)
{
    initialize_bfs_offsets();
}

void SensorModel::initialize_bfs_offsets()
{
    /// TODO: Initialize the BFS offsets based on the search range 
    
}

double SensorModel::likelihood(const mbot_lcm_msgs::particle_t& sample, 
                               const mbot_lcm_msgs::lidar_t& scan, 
                               const OccupancyGrid& map)
{
    /// TODO: Compute the likelihood of the given particle using the provided laser scan and map. 
    double scanScore = 1.0;
    MovingLaserScan movingScan(scan, sample.parent_pose, sample.pose);

    const double fraction_factor = 0.5;

    for(auto& ray : movingScan)
    {
        Point<double> endpoint(ray.origin.x + ray.range * std::cos(ray.theta), ray.origin.y + ray.range * std::sin(ray.theta));
        auto endpoint_ingrid = global_position_to_grid_cell(endpoint, map);
        // std::cout << map.logOdds(endpoint_ingrid.x, endpoint_ingrid.y) << std::endl;
        if(map.logOdds(endpoint_ingrid.x, endpoint_ingrid.y) > 0)
        {
            scanScore += 1.0;
            // scanScore += map.logOdds(endpoint_ingrid.x, endpoint_ingrid.y);
            // std::cout << "Entered if block: logOdds value is " << map.logOdds(endpoint_ingrid.x, endpoint_ingrid.y) << std::endl;        
        }

        // Check the cell at the endpoint
        // double logOddsAtEndpoint = map.logOdds(endpoint_ingrid.x, endpoint_ingrid.y);
        // if(logOddsAtEndpoint > 0)
        // {
        //     scanScore += logOddsAtEndpoint;

        // }
        // else
        // {
        //     // If the endpoint is not a hit, check the cells along the ray path
        //     double fractionalScore = 0.0;

        //     // Check the cell before the endpoint
        //     // Point<double> beforeEndpoint(ray.origin.x + (ray.range - 1) * std::cos(ray.theta)* map.cellsPerMeter(), 
        //     //                              ray.origin.y + (ray.range - 1) * std::sin(ray.theta)* map.cellsPerMeter());
        //     Point<double> beforeEndpoint(ray.origin.x + (ray.range - 1) * std::cos(ray.theta), 
        //                                  ray.origin.y + (ray.range - 1) * std::sin(ray.theta));
        //     Point<double> beforeEndpoint_ingrid = global_position_to_grid_cell(beforeEndpoint, map);

        //     fractionalScore += std::max(0.0, static_cast<double>(map.logOdds(beforeEndpoint_ingrid.x, beforeEndpoint_ingrid.y)));

        //     // Check the cell after the endpoint
        //     // Point<double> afterEndpoint(ray.origin.x + (ray.range + 1) * std::cos(ray.theta)* map.cellsPerMeter(), 
        //     //                             ray.origin.y + (ray.range + 1) * std::sin(ray.theta)* map.cellsPerMeter());
        //     Point<double> afterEndpoint(ray.origin.x + (ray.range + 1) * std::cos(ray.theta), 
        //                                 ray.origin.y + (ray.range + 1) * std::sin(ray.theta));
        //     Point<double> afterEndpoint_ingrid = global_position_to_grid_cell(afterEndpoint, map);

        //     fractionalScore += std::max(0.0, static_cast<double>(map.logOdds(afterEndpoint_ingrid.x, afterEndpoint_ingrid.y)));

        //     // Add a fraction of the log odds
        //     scanScore += fractionalScore * fraction_factor;
        // }


        // for(adjusted_ray_t ray:movingScan)
        // {
        //     scanScore += map.logOdds(rayEnd.x, rayEnd.y);
        // }
        // else
        // {
        //     Point<int> nearestOccupiedCell = gridBFS(rayEnd, map);
        //     Point<double> nearestOccupiedCellGlobal = grid_position_to_global_position(nearestOccupiedCell, map);
        //     double distance = sqrt(pow(endpoint.x - nearestOccupiedCellGlobal.x, 2) + pow(endpoint.y - nearestOccupiedCellGlobal.y, 2));
        //     scanScore += NormalPdf(distance);
        // }
    }
    return scanScore; // Placeholder
    
}

// double SensorModel::scoreRay(const adjusted_ray_t& ray, const OccupancyGrid& map)
// {
//     /// TODO: Compute a score for a given ray based on its end point and the map. 
//     // Consider the offset from the nearest occupied cell.  
//     Point<double> endpoint(ray.origin.x + ray.range*cos(ray.theta), ray.origin.y + ray.range*sin(ray.theta));

    
// }

double SensorModel::NormalPdf(const double& x)
{
    return (1.0/(sqrt(2.0 * M_PI)*sigma_hit_))*exp((-0.5*x*x)/(sigma_hit_*sigma_hit_));
}

Point<int> SensorModel::gridBFS(const Point<int> end_point, const OccupancyGrid& map)
{
    /// TODO: Use Breadth First Search to find the nearest occupied cell to the given end point. 
    return Point<int>(0,0); // Placeholder
    
}

Point<float> SensorModel::getRayEndPointOnMap(const adjusted_ray_t& ray, const OccupancyGrid& map)
{
    /// TODO: Calculate the end point of a given ray on the map 
    return Point<float>(0.0f, 0.0f); // Placeholder
    
}
