# import pygame
# from pygame.locals import *
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
    while time_elapsed < duration - dt:
        lc.publish("MBOT_VEL_CMD", command.encode())
        time.sleep(dt)
        time_elapsed += dt
    # lc.publish("MBOT_VEL_CMD", command.encode())
    # time.sleep(duration)
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
    while time_elapsed < time_to_cover - dt:
        lc.publish("MBOT_VEL_CMD", command.encode())
        time.sleep(dt)
        time_elapsed += dt
    # lc.publish("MBOT_VEL_CMD", command.encode())
    # time.sleep(time_to_cover)
    stop_robot()

if __name__ == "__main__":
    # move_straight(0.25, 2)
    move_straight_for_distance(0.70,2)
    turn_around(np.pi/4, 2.65)
    move_straight_for_distance(1.3,4)
    turn_around(-np.pi/4, 2.65)
    move_straight_for_distance(0.65,2)
    turn_around(-np.pi/4, 2.65)
    move_straight_for_distance(0.65,2)
    turn_around(np.pi/4, 2.65)
    move_straight_for_distance(0.65,2)