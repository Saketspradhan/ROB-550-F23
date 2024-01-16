#include <utils/grid_utils.hpp>
#include <slam/particle_filter.hpp>
#include <slam/occupancy_grid.hpp>
#include <mbot_lcm_msgs/pose2D_t.hpp>
#include <mbot_lcm_msgs/particle_t.hpp>
#include <utils/geometric/angle_functions.hpp>
#include <cassert>
#include <iostream>
#include <queue>


ParticleFilter::ParticleFilter(int numParticles)
: kNumParticles_ (numParticles),
  samplingAugmentation_(0.5, 0.9, numParticles),
  distribution_quality(1),
  quality_reinvigoration_percentage(0.1)
{
    assert(kNumParticles_ > 1);
    posterior_.resize(kNumParticles_);
}


void ParticleFilter::initializeFilterAtPose(const mbot_lcm_msgs::pose2D_t& pose)
{
    ///////////// TODO: Implement your method for initializing the particles in the particle filter /////////////////
    double sample_weight = 1.0 / kNumParticles_;
    posteriorPose_ = pose;

    for(auto&& p : posterior_)
    {
        p.pose.x = posteriorPose_.x;
        p.pose.y = posteriorPose_.y;
        p.pose.theta = wrap_to_pi(posteriorPose_.theta);
        p.weight = sample_weight;
        p.pose.utime = pose.utime;
        p.parent_pose = p.pose;
    }

}

void ParticleFilter::initializeFilterRandomly(const OccupancyGrid& map)
{
    ///////////// TODO: Implement your method for initializing the particles in the particle filter /////////////////
    double sample_weight = 1.0 / kNumParticles_;
    std::random_device rd;
    std::mt19937 generator(rd());

    mbot_lcm_msgs::occupancy_grid_t grid = map.toLCM();
    double x_min = grid.origin_x - (grid.width * grid.meters_per_cell) / 2.0;
    double x_max = grid.origin_x + (grid.width * grid.meters_per_cell) / 2.0;
    double y_min = grid.origin_y - (grid.height * grid.meters_per_cell) / 2.0;
    double y_max = grid.origin_y + (grid.height * grid.meters_per_cell) / 2.0;

    std::uniform_real_distribution<> dis_x(x_min, x_max);
    std::uniform_real_distribution<> dis_y(y_min, y_max);
    std::uniform_real_distribution<> dis_theta(-M_PI, M_PI);

    for (auto&& particle : posterior_) {
        particle.pose.x = dis_x(generator);
        particle.pose.y = dis_y(generator);
        
        particle.pose.theta = dis_theta(generator);
        particle.weight = sample_weight;
        // Set other necessary fields for each particle
    }

}

void ParticleFilter::resetOdometry(const mbot_lcm_msgs::pose2D_t& odometry)
{
    actionModel_.resetPrevious(odometry);
}


mbot_lcm_msgs::pose2D_t ParticleFilter::updateFilter(const mbot_lcm_msgs::pose2D_t& odometry,
                                                        const mbot_lcm_msgs::lidar_t& laser,
                                                        const OccupancyGrid& map)
{
    bool hasRobotMoved = actionModel_.updateAction(odometry);

    if(hasRobotMoved)
    {
        auto prior = resamplePosteriorDistribution();
        // reinvigoratePriorDistribution(prior);
        auto proposal = computeProposalDistribution(prior);
        posterior_ = computeNormalizedPosterior(proposal, laser, map);
    }
    posteriorPose_ = estimatePosteriorPose(posterior_);



    // // Testing:
    // auto printParticleList = [](const ParticleList& list, const std::string& name) {
    //     std::cout << name << ": [" << std::endl;
    //     for (const auto& particle : list) {
    //         std::cout << "    Particle: { Pose: {" 
    //                   << "x: " << particle.pose.x << ", y: " << particle.pose.y << ", theta: " << particle.pose.theta 
    //                   << "}, Weight: " << particle.weight << " }" << std::endl;
    //     }
    //     std::cout << "]" << std::endl;
    // };

    // printParticleList(prior, "Prior");
    // printParticleList(proposal, "Proposal");
    // printParticleList(posterior_, "Posterior");

    // // Print the pose2D_t contents
    // std::cout << "PosteriorPose: { "
    //           << "x: " << posteriorPose_.x << ", y: " << posteriorPose_.y 
    //           << ", theta: " << posteriorPose_.theta << ", utime: " << posteriorPose_.utime 
    //           << " }" << std::endl;


    posteriorPose_.utime = odometry.utime;

    return posteriorPose_;
}

mbot_lcm_msgs::pose2D_t ParticleFilter::updateFilterActionOnly(const mbot_lcm_msgs::pose2D_t& odometry)
{
    // Only update the particles if motion was detected. If the robot didn't move, then
    // obviously don't do anything.
    bool hasRobotMoved = actionModel_.updateAction(odometry);

    if(hasRobotMoved)
    {
        auto prior = resamplePosteriorDistribution();
        auto proposal = computeProposalDistribution(prior);
        posterior_ = proposal;
    }

    posteriorPose_ = estimatePosteriorPose(posterior_);

    return posteriorPose_;
}


mbot_lcm_msgs::pose2D_t ParticleFilter::poseEstimate(void) const
{
    return posteriorPose_;
}


mbot_lcm_msgs::particles_t ParticleFilter::particles(void) const
{
    mbot_lcm_msgs::particles_t particles;
    particles.num_particles = posterior_.size();
    particles.particles = posterior_;
    return particles;
}


ParticleList ParticleFilter::resamplePosteriorDistribution(const OccupancyGrid& map,
                                                           const bool keep_best,
                                                           const bool reinvigorate)
{
    //////////// TODO: Implement your algorithm for resampling from the posterior distribution ///////////////////
    // ParticleList resampled;
    // double sample_weight = 1.0 / kNumParticles_;
    // std::random_device rd;
    // std::mt19937 generator(rd());
    // std::uniform_real_distribution<> dis(0.0, sample_weight);

    // double r = dis(generator);
    // double c = posterior_[0].weight;
    // int i = 0;

    // for (int m = 0; m < kNumParticles_; m++)
    // {
    //     double U = r + m * sample_weight;
    //     while (U > c)
    //     {
    //         i++;
    //         c += posterior_[i].weight;
    //     }
    //     resampled.push_back(posterior_[i]);
    // }
    
    // return resampled;  // Placeholder
    ParticleList samples;

    if (kNumParticles_ < 1 || posterior_.size() < 1) return samples;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<float> distribution(0.0, 1.0 / kNumParticles_);

    float r = distribution(gen);
    size_t idx = 0;
    float s = posterior_[idx].weight;

    for (int i = 0; i < kNumParticles_; ++i)
    {
        float u = r + i * 1. / kNumParticles_;
        while (u > s) {
            ++idx;
            s += posterior_[idx].weight;
        }
        samples.push_back(posterior_[idx]);
    }

    return samples;
}


ParticleList ParticleFilter::resamplePosteriorDistribution(const bool keep_best,
                                                           const bool reinvigorate)
{
    //////////// TODO: Implement your algorithm for resampling from the posterior distribution ///////////////////
    // ParticleList resampled;
    // double sample_weight = 1.0 / kNumParticles_;
    // std::random_device rd;
    // std::mt19937 generator(rd());
    // std::uniform_real_distribution<> dis(0.0, sample_weight);

    // double r = dis(generator);
    // double c = posterior_[0].weight;
    // int i = 0;

    // for (int m = 0; m < kNumParticles_; m++)
    // {
    //     double U = r + m * sample_weight;
    //     while (U > c)
    //     {
    //         i++;
    //         c += posterior_[i].weight;
    //     }
    //     resampled.push_back(posterior_[i]);
    // }
    
    // return resampled;  // Placeholder
    ParticleList samples;

    if (kNumParticles_ < 1 || posterior_.size() < 1) return samples;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<float> distribution(0.0, 1.0 / kNumParticles_);

    float r = distribution(gen);
    size_t idx = 0;
    float s = posterior_[idx].weight;

    for (int i = 0; i < kNumParticles_; ++i)
    {
        float u = r + i * 1. / kNumParticles_;
        while (u > s) {
            ++idx;
            s += posterior_[idx].weight;
        }
        samples.push_back(posterior_[idx]);
    }

    return samples;
}


void ParticleFilter::reinvigoratePriorDistribution(ParticleList& prior)
{
    // Augmentation: if sensor model suspects an average particle quality of
    //      less than 15%, invigorate
    if (distribution_quality < 0.15)  // TODO: make 0.15 a parameter
    {
        int count = 0;
        int max_count = floor(quality_reinvigoration_percentage * prior.size());

        std::random_device rd;
        std::default_random_engine generator(rd());
        auto ud01 = std::uniform_real_distribution<double>(0.0, 1.0);
        int step = std::max<int>(1, floor(ud01(generator) * prior.size() / max_count));

        for (int i = 0; i < max_count; i++)
        {
            prior[i*step] = randomPoseGen_.get_particle();
        }

    }

    // // Augmentation: randomize any unreasonable samples
    // if(map != nullptr)
    // {
    //     for (int i = 0; i < prior.size(); i++)
    //     {
    //         const auto& p = prior[i].pose;
    //         if(!map->isCellInGrid(p.x, p.y) ||
    //           (map->isCellInGrid(p.x, p.y) && (map->logOdds(p.x, p.y) >= 0)))
    //         {
    //             std::cout << "\tinvalid sample!!" << ", "
    //                 << !map->isCellInGrid(p.x, p.y) << ", "
    //                 << (map->isCellInGrid(p.x, p.y) && (map->logOdds(p.x, p.y) >= 0.0))
    //                 << std::endl;


    //             prior[i] = randomPoseGen_.get_particle();
    //         }
    //     }
    // }
}


ParticleList ParticleFilter::computeProposalDistribution(const ParticleList& prior)
{
    //////////// TODO: Implement your algorithm for creating the proposal distribution by sampling from the ActionModel
    ParticleList proposal;
    for(auto&& p : prior)
    {
        proposal.push_back(actionModel_.applyAction(p));
    }
    
    return proposal;  // Placeholder
}


ParticleList ParticleFilter::computeNormalizedPosterior(const ParticleList& proposal,
                                                        const mbot_lcm_msgs::lidar_t& laser,
                                                        const OccupancyGrid& map)
{
    /////////// TODO: Implement your algorithm for computing the normalized posterior distribution using the
    ///////////       particles in the proposal distribution
    ParticleList posterior;
    double sumWeights = 0.0;

    for (auto&& particle : proposal) {
        mbot_lcm_msgs::particle_t weighted = particle;
        weighted.weight = sensorModel_.likelihood(particle, laser, map);

        if(std::isnan(weighted.weight) || std::isinf(weighted.weight)) {
            std::cerr << "Warning: Weight became NaN or Inf" << std::endl;
            // Handle the invalid weight case, e.g., by skipping this particle or assigning a default weight
            continue; 
        }
        
        sumWeights += weighted.weight;

        if(std::isnan(sumWeights) || std::isinf(sumWeights)) {
            std::cerr << "Error: sumWeights became NaN or Inf" << std::endl;
            // Decide how to handle this error, possibly by aborting the function
            return posterior; // Or some other error handling
        }

        posterior.push_back(weighted);
    }

    // Normalize the weights
    for (auto&& particle : posterior) {
        particle.weight /= sumWeights;
    }

    return posterior;  // Placeholder
}

struct Compare_Particle
{
    bool operator() (mbot_lcm_msgs::particle_t p1, mbot_lcm_msgs::particle_t p2)
    {
        return (p1.weight < p2.weight); // less than for max pq
    }
};

mbot_lcm_msgs::pose2D_t ParticleFilter::estimatePosteriorPose(const ParticleList& posterior)
{
    //////// TODO: Implement your method for computing the final pose estimate based on the posterior distribution
    // Figure out which pose to take for the posterior pose
    // Weighted average is simple, but could be very bad
    // Maybe only take the best x% and then average.


    mbot_lcm_msgs::pose2D_t pose;
    
    double total_weight = 0.0;
    double x_sum = 0.0, y_sum = 0.0, theta_x_sum = 0.0, theta_y_sum = 0.0;

    std::priority_queue<mbot_lcm_msgs::particle_t, std::vector<mbot_lcm_msgs::particle_t>, Compare_Particle> pq;

    for (const auto& particle : posterior) {
        pq.push(particle);
    }

    mbot_lcm_msgs::particle_t particle;
    for (int i = 0; i < posterior.size() / 15; i++) {

    // for (auto&& particle: posterior){
        particle = pq.top();
        pq.pop();
        x_sum += particle.pose.x * particle.weight;
        y_sum += particle.pose.y * particle.weight;
        theta_x_sum += std::cos(particle.pose.theta) * particle.weight;
        theta_y_sum += std::sin(particle.pose.theta) * particle.weight;
        total_weight += particle.weight;
    }

    pose.x = x_sum / total_weight;
    pose.y = y_sum / total_weight;

    theta_x_sum /= total_weight;
    theta_y_sum /= total_weight;

    pose.theta = atan2(theta_y_sum, theta_x_sum);

    return pose;
}


mbot_lcm_msgs::pose2D_t ParticleFilter::computeParticlesAverage(const ParticleList& particles_to_average)
{
    mbot_lcm_msgs::pose2D_t avg_pose;
    avg_pose.x = 0.0;
    avg_pose.y = 0.0;
    avg_pose.theta = 0.0;
    double sum_weight = 0.0;

    // Aux variables to compute theta average
    double theta_x = 0.0;
    double theta_y = 0.0;
    for (auto &&p : particles_to_average)
    {
        avg_pose.x += p.weight * p.pose.x;
        avg_pose.y += p.weight * p.pose.y;
        theta_x += p.weight * std::cos(p.pose.theta);
        theta_y += p.weight * std::sin(p.pose.theta);

        sum_weight += p.weight;
    }
    avg_pose.x /= sum_weight;
    avg_pose.y /= sum_weight;
    theta_x /= sum_weight;
    theta_y /= sum_weight;
    avg_pose.theta = std::atan2(theta_y, theta_x);

    return avg_pose;
}