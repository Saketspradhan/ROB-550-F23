# import pygame
# from pygame.locals import *

# sudo systemctl stop mbot-motion-controller.service

import time
import numpy as np
import lcm
import sys
sys.path.append('/usr/lib/python3.9/site-packages/')
from mbot_lcm_msgs.mbot_motor_vel_t import mbot_motor_vel_t

from mbot_lcm_msgs.twist2D_t import twist2D_t

LIN_VEL_CMD = 100.0 # rad/s
ANG_VEL_CMD = 50.0 # rad/s

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")

def move_straight(speed, duration):
    command = twist2D_t()
    command.vx = speed
    command.vy = 0
    command.wz = 0
    lc.publish("MBOT_VEL_CMD", command.encode())
    time.sleep(duration)
    stop_robot()

def turn_around(turn_speed, duration):
    command = twist2D_t()
    command.vx = 0
    command.vy = 0
    command.wz = turn_speed

    dt = 0.01
    time_elapsed = 0
    lc.publish("MBOT_VEL_CMD", command.encode())
    time.sleep(duration)
    stop_robot()

def stop_robot():
    command = twist2D_t()
    command.vx = 0
    command.vy = 0
    command.wz = 0
    lc.publish("MBOT_VEL_CMD", command.encode())
    time.sleep(0.5)

def move_straight_for_distance(distance, time_to_cover):
    speed = distance / time_to_cover

    command = twist2D_t()
    command.vx = speed
    command.vy = 0
    command.wz = 0

    dt = 0.01
    time_elapsed = 0

    lc.publish("MBOT_VEL_CMD", command.encode())
    time.sleep(time_to_cover)
    stop_robot()

if __name__ == "__main__":
    # move_straight_for_distance(0.7,2)
    # turn_around(np.pi/4, 2)
    # move_straight_for_distance(1.3,4)
    # turn_around(-np.pi/4, 2)
    # move_straight_for_distance(0.65,2)
    # turn_around(-np.pi/4, 2)
    # move_straight_for_distance(0.65,2)
    # turn_around(np.pi/4, 2)
    # move_straight_for_distance(0.65,2)

    # SLOW
    # move_straight_for_distance(0.61,2)
    # turn_around(-np.pi/4.1, 2)
    # move_straight_for_distance(0.61,2.2)
    # turn_around(np.pi/4.4, 2)
    # move_straight_for_distance(0.61,2)
    # turn_around(np.pi/4.2, 2)
    # move_straight_for_distance(1.22,5)
    # turn_around(-np.pi/4, 2)
    # move_straight_for_distance(0.61,2)
    # turn_around(-np.pi/4, 2)
    # move_straight_for_distance(1.22,5.5)
    # turn_around(np.pi/4, 2)
    # move_straight_for_distance(0.61,2)
    # turn_around(np.pi/4.2, 2)
    # move_straight_for_distance(0.61,2)
    # turn_around(-np.pi/4, 2)
    # move_straight_for_distance(0.78,2)

    # FAST
    # move_straight_for_distance(0.61,0.85)
    # turn_around(-np.pi/1.15, 0.5)
    # move_straight_for_distance(0.61,0.85)
    # turn_around(np.pi/1.35, 0.6)
    # move_straight_for_distance(0.55,0.75)
    # turn_around(np.pi/1.36, 0.6)
    # move_straight_for_distance(1.1,1.8)
    # turn_around(-np.pi/1.25, 0.6)
    # move_straight_for_distance(0.64,0.83)
    # turn_around(-np.pi/1.38, 0.58)
    # move_straight_for_distance(1.14,1.7)
    # turn_around(np.pi/1.18, 0.6)
    # move_straight_for_distance(0.61,0.85)
    # turn_around(np.pi/1.42, 0.6)
    # move_straight_for_distance(0.5,0.85)
    # turn_around(-np.pi/1.2, 0.6)
    # move_straight_for_distance(0.74,1.05)

    # move_straight_for_distance(0.61,1.2)
    # turn_around(-np.pi/4, 2)
    # move_straight_for_distance(0.61,1.2)
    # turn_around(np.pi/4.2, 2)
    # move_straight_for_distance(0.61,1.2)
    # turn_around(np.pi/4.2, 2)
    # move_straight_for_distance(1.22,3.5)
    # turn_around(-np.pi/4, 2)
    # move_straight_for_distance(0.61,1.2)
    # turn_around(-np.pi/4.2, 2)
    # move_straight_for_distance(1.22,3.5)
    # turn_around(np.pi/4.2, 2)
    # move_straight_for_distance(0.61,1.2)
    # turn_around(np.pi/4.5, 2)
    # move_straight_for_distance(0.61,1.2)
    # turn_around(-np.pi/3.8, 2)
    # move_straight_for_distance(0.78,1.2)

    # move_straight_for_distance(1.22,2)
    # turn_around(np.pi/4, 2.65)
    # move_straight_for_distance(1.22,2)

    # SQUARE
    move_straight_for_distance(1.0541,2)
    # turn_around(np.pi/4, 2)
    # move_straight_for_distance(0.65,2)
    # turn_around(np.pi/4, 2)
    # move_straight_for_distance(0.65,2)
    # turn_around(np.pi/4, 2)
    # move_straight_for_distance(0.65,2)
    # turn_around(np.pi/4, 2)
