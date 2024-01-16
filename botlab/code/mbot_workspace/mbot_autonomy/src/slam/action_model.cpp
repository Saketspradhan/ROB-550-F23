#include <slam/action_model.hpp>
#include <mbot_lcm_msgs/particle_t.hpp>
#include <utils/geometric/angle_functions.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>


ActionModel::ActionModel(void)
: k1_(0.001f)
, k2_(0.001f)
, min_dist_(0.025)
, min_theta_(0.02)
, initialized_(false)
{
    //////////////// TODO: Handle any initialization for your ActionModel /////////////////////////
    std::random_device rd;  
    numberGenerator_ = std::mt19937(rd());

}


void ActionModel::resetPrevious(const mbot_lcm_msgs::pose2D_t& odometry)
{
    previousPose_ = odometry;
}


bool ActionModel::updateAction(const mbot_lcm_msgs::pose2D_t& odometry)
{
    ////////////// TODO: Implement code here to compute a new distribution of the motion of the robot ////////////////
    if(!initialized_)
    {
        previousPose_ = odometry;
        initialized_ = true;
        return false;
    }

    dx_ = odometry.x - previousPose_.x;
    dy_ = odometry.y - previousPose_.y;
    dtheta_ = odometry.theta - previousPose_.theta;

    trans_ = sqrt(dx_*dx_ + dy_*dy_);
    float theta = atan2(dy_, dx_);
    rot1_ = angle_diff(theta, previousPose_.theta);

    float direction = 1.0;

    if(std::abs(rot1_) > M_PI/2.0){
        rot1_ = angle_diff(M_PI, rot1_ );
        direction = -1.0;
    }

    rot2_ = angle_diff(dtheta_, rot1_);

    // moved_ = (trans_ > min_dist_) || (fabs(rot2_) > min_theta_);
    moved_ = (dx_ != 0.0) || (dy_ != 0.0)|| (dtheta_ != 0.0);

    if (moved_)
    {
        rot1Std_ = std::sqrt(k1_ * std::abs(rot1_));
        transStd_ = std::sqrt(k2_ * std::abs(trans_));
        rot2Std_ = std::sqrt(k1_ * std::abs(rot2_));
    }

    trans_ *= direction;
    previousPose_ = odometry;
    utime_ = odometry.utime;

    return moved_;    // Placeholder
}

mbot_lcm_msgs::particle_t ActionModel::applyAction(const mbot_lcm_msgs::particle_t& sample)
{
    ////////////// TODO: Implement your code for sampling new poses from the distribution computed in updateAction //////////////////////
    // Make sure you create a new valid particle_t. Don't forget to set the new time and new parent_pose.
    mbot_lcm_msgs::particle_t newSample = sample;
    
    float sampledRot1 = std::normal_distribution<float>(rot1_, rot1Std_)(numberGenerator_);
    float sampledTrans = std::normal_distribution<float>(trans_, transStd_)(numberGenerator_);
    float sampledRot2 = std::normal_distribution<float>(rot2_, rot2Std_)(numberGenerator_);

    newSample.pose.x = sample.pose.x + sampledTrans * cos(sample.pose.theta + sampledRot1);
    newSample.pose.y = sample.pose.y + sampledTrans * sin(sample.pose.theta + sampledRot1);
    newSample.pose.theta = wrap_to_pi(sample.pose.theta + sampledRot1 + sampledRot2);
    newSample.parent_pose = sample.pose;
    newSample.pose.utime = utime_;

    return newSample;
}
