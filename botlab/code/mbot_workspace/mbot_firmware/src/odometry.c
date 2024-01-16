#include "odometry.h"
#include <math.h>

int mbot_calculate_diff_body_vel(float wheel_left_vel, float wheel_right_vel, serial_twist2D_t *mbot_vel){
    mbot_vel->vx =  DIFF_WHEEL_RADIUS * (wheel_left_vel - wheel_right_vel) / 2.0f;
    mbot_vel->vy = 0;
    mbot_vel->wz =  DIFF_WHEEL_RADIUS * (-wheel_left_vel - wheel_right_vel) / (2.0f * DIFF_BASE_RADIUS);
    return 0; // Return 0 to indicate success
}
int mbot_calculate_diff_body_vel_imu(float wheel_left_vel, float wheel_right_vel, serial_mbot_imu_t imu, serial_twist2D_t *mbot_vel, float *theta_prev, float *odo_theta_prev, float *gyro_theta_prev){

    /*

    Input:

    Encoder-Read Velocities: wheel_left_vel and wheel_right_vel
    Imu Data: imu
    Pointer to Estimated Velocities: mbot_vel

    */

    mbot_vel->vx = DIFF_WHEEL_RADIUS * (wheel_left_vel - wheel_right_vel) / 2.0f;
    mbot_vel->vy = 0;

    /* TODO: IMU Fusion For Better Angular Velocity */

    

    // Implement Gyrodometry or Filter Here

    // float odo_theta = DIFF_WHEEL_RADIUS * (-wheel_left_vel - wheel_right_vel) / (2.0f * DIFF_BASE_RADIUS) * MAIN_LOOP_PERIOD;
    float gyro_theta = imu.angles_rpy[2];

    // float delta_odo_theta = odo_theta - *odo_theta_prev;
    float delta_odo_theta = DIFF_WHEEL_RADIUS * (-wheel_left_vel - wheel_right_vel) / (2.0f * DIFF_BASE_RADIUS) * MAIN_LOOP_PERIOD;
    float delta_gyro_theta = gyro_theta - *gyro_theta_prev;

    if (gyro_theta < 0 && *gyro_theta_prev > 0) {
        if (gyro_theta < -1.5) {
            // gyro_theta ~ -3.14 and gyro_theta_prev ~ 3.14
            delta_gyro_theta = (M_PI - *gyro_theta_prev) + (gyro_theta - (-M_PI));
        }else{
            gyro_theta - *gyro_theta_prev;
        }
    }

    if (gyro_theta > 0 && *gyro_theta_prev < 0) {
        if (gyro_theta > 1.5) {
            // gyro_theta ~ 3.14 and gyro_theta_prev ~ -3.14
            delta_gyro_theta = (-M_PI - *gyro_theta_prev) + (gyro_theta - M_PI);
        }else{
            delta_gyro_theta = gyro_theta - *gyro_theta_prev;
        }
    }

    float imu_wz = delta_gyro_theta / MAIN_LOOP_PERIOD;


    // *odo_theta_prev = odo_theta;
    *gyro_theta_prev = gyro_theta;

    float theta = 0.0f;

    if (fabs(delta_gyro_theta - delta_odo_theta) > 0.1) {
        theta = *theta_prev + (delta_gyro_theta);
    } else{
        theta = *theta_prev + (delta_odo_theta);
    }

    // if (theta < -M_PI) {
        
    // }

    mbot_vel->wz = (theta - *theta_prev) / MAIN_LOOP_PERIOD;

    mbot_vel->wz = imu_wz;

    *theta_prev = theta;

    return 0; // Return 0 to indicate success
}
int mbot_calculate_omni_body_vel(float wheel_left_vel, float wheel_right_vel, float wheel_back_vel, serial_twist2D_t *mbot_vel){
    mbot_vel->vx =  OMNI_WHEEL_RADIUS * (wheel_left_vel * INV_SQRT3 - wheel_right_vel * INV_SQRT3);
    mbot_vel->vy =  OMNI_WHEEL_RADIUS * (-wheel_left_vel / 3.0 - wheel_right_vel / 3.0 + wheel_back_vel * (2.0 / 3.0));
    mbot_vel->wz =  OMNI_WHEEL_RADIUS * -(wheel_left_vel + wheel_right_vel + wheel_back_vel) / (3.0f * OMNI_BASE_RADIUS);
    return 0; // Return 0 to indicate success
}
int mbot_calculate_omni_body_vel_imu(float wheel_left_vel, float wheel_right_vel, float wheel_back_vel, serial_mbot_imu_t imu, serial_twist2D_t *mbot_vel){
    return 0; // Return 0 to indicate success
}

int mbot_calculate_odometry(serial_twist2D_t mbot_vel, float dt, serial_pose2D_t *odometry){
    float vx_space = mbot_vel.vx * cos(odometry->theta) - mbot_vel.vy * sin(odometry->theta);
    float vy_space = mbot_vel.vx * sin(odometry->theta) + mbot_vel.vy * cos(odometry->theta);

    odometry->x += vx_space * dt;
    odometry->y += vy_space * dt;
    odometry->theta += mbot_vel.wz * dt; // Uses the Measured Velocity

    // Normalize theta to be between -pi and pi
    while (odometry->theta > M_PI) odometry->theta -= 2 * M_PI;
    while (odometry->theta <= -M_PI) odometry->theta += 2 * M_PI;

    return 0; // Return 0 to indicate success
}
