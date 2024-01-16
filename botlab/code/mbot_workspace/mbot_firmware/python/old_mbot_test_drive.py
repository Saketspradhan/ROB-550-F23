# import pygame
# from pygame.locals import *
import time
import numpy as np
import lcm
import sys
sys.path.append('/usr/lib/python3.9/site-packages/')
from mbot_lcm_msgs.twist2D_t import twist2D_t

LIN_VEL_CMD = 100.0 # rad/s
ANG_VEL_CMD = 50.0 # rad/s

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
# pygame.init()
# pygame.display.set_caption("MBot TeleOp")
# screen = pygame.display.set_mode([100,100])
# pygame.key.set_repeat(5)
time.sleep(0.5)
running = True

print("Starting to move")

command = twist2D_t()
command.vx = 0 # 0.5
command.vy = 0
command.wz = 3.1415/4
lc.publish("MBOT_VEL_CMD", command.encode())
time.sleep(8)

print("Starting to stop")

command = twist2D_t()
command.vx = 0
command.vy = 0
command.wz = 0
lc.publish("MBOT_VEL_CMD", command.encode())
time.sleep(0.5)

# fwd_vel = 0.1
# turn_vel = 0.0
# command = mbot_motor_vel_t()
# command.velocity[0] = -fwd_vel + turn_vel
# command.velocity[1] = fwd_vel - turn_vel
# lc.publish("MBOT_MOTOR_VEL_CMD",command.encode())
# time.sleep(0.4)

# fwd_vel = 0.25
# turn_vel = 0.0
# command = mbot_motor_vel_t()
# command.velocity[0] = -fwd_vel + turn_vel
# command.velocity[1] = fwd_vel - turn_vel
# lc.publish("MBOT_MOTOR_VEL_CMD",command.encode())
# time.sleep(0.4)

# fwd_vel = 0.4
# turn_vel = 0.0
# command = mbot_motor_vel_t()
# command.velocity[0] = -fwd_vel + turn_vel
# command.velocity[1] = fwd_vel - turn_vel
# lc.publish("MBOT_MOTOR_VEL_CMD",command.encode())
# time.sleep(2.3)


# fwd_vel = 0.25
# turn_vel = 0.0
# command = mbot_motor_vel_t()
# command.velocity[0] = -fwd_vel + turn_vel
# command.velocity[1] = fwd_vel - turn_vel
# lc.publish("MBOT_MOTOR_VEL_CMD",command.encode())
# time.sleep(0.4)

# fwd_vel = 0.1
# turn_vel = 0.0
# command = mbot_motor_vel_t()
# command.velocity[0] = -fwd_vel + turn_vel
# command.velocity[1] = fwd_vel - turn_vel
# lc.publish("MBOT_MOTOR_VEL_CMD",command.encode())
# time.sleep(0.4)


# fwd_vel = 0.20
# turn_vel = 0.0
# command = mbot_motor_vel_t()
# command.velocity[0] = -fwd_vel + turn_vel
# command.velocity[1] = fwd_vel - turn_vel
# lc.publish("MBOT_MOTOR_VEL_CMD",command.encode())
# time.sleep(2.0)

# fwd_vel = 0.35
# turn_vel = 0.0
# command = mbot_motor_vel_t()
# command.velocity[0] = -fwd_vel + turn_vel
# command.velocity[1] = fwd_vel - turn_vel
# lc.publish("MBOT_MOTOR_VEL_CMD",command.encode())
# time.sleep(2.0)

# fwd_vel = 0.0
# turn_vel = 0.25
# command = mbot_motor_vel_t()
# command.velocity[0] = -fwd_vel + turn_vel
# command.velocity[1] = fwd_vel + turn_vel
# lc.publish("MBOT_MOTOR_VEL_CMD",command.encode())
# time.sleep(1)

# fwd_vel = 0.3
# turn_vel = 0.0
# command = mbot_motor_vel_t()
# command.velocity[0] = -fwd_vel + turn_vel
# command.velocity[1] = fwd_vel - turn_vel
# lc.publish("MBOT_MOTOR_VEL_CMD",command.encode())
# time.sleep(2.8)

# print('going here')

# command.velocity[0] = 0.0
# command.velocity[1] = 0.0
# lc.publish("MBOT_MOTOR_VEL_CMD",command.encode())
