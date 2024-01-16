# import pygame
# from pygame.locals import *

import time 
import matplotlib
matplotlib.use('Agg')  # This needs to be done before importing pyplot
import matplotlib.pyplot as plt

import numpy as np
import lcm
import sys
sys.path.append('/usr/lib/python3.9/site-packages/')
from mbot_lcm_msgs.twist2D_t import twist2D_t

cmds = []
errors = []
headings = []

def my_handler(channel, data):
    msg = twist2D_t.decode(data)
    # print("Received message on channel \"%s\"" % channel)
    # print("   vy   = %s" % str(msg.vy))
    # print("")
    errors.append(float(msg.vy))
    cmds.append(float(msg.wz))
    headings.append(float(msg.vx))
    # if (abs(float(msg.vy)) < 2.0 ):
    #     errors.append(float(msg.vy))
    #     cmds.append(float(msg.wz))
    #     headings.append(float(msg.vx))
    # else:
    #     print("Error Thetas:")
    #     print(headings[-5:-1])
    #     print(msg.vx)
    #     print("Error Measured w:")
    #     print(errors[-5:-1])
    #     print(msg.vy)


LIN_VEL_CMD = 100.0 # rad/s
ANG_VEL_CMD = 50.0 # rad/s

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")

time.sleep(0.5)
running = True

print("Starting to move")

# Comment the w and v low pass and PID filters when tuning the wheel speed PIDs!

command = twist2D_t()
command.vx = 0
command.vy = 0
command.wz = (np.pi/4)
lc.publish("MBOT_VEL_CMD", command.encode())


subscription = lc.subscribe("MBOT_VEL", my_handler)

try:
    while True:
        try:
            lc.handle_timeout(100)  # Handle LCM events for 100 milliseconds
        except OSError as e:
            print(f"An OS error occurred: {e}")
            # Decide how to handle the OSError here. You may choose to continue or break the loop.
            # For example, if you want to continue, just pass; if you want to exit, use break.
            pass
except KeyboardInterrupt:
    print("Stopping due to KeyboardInterrupt...")

    # Stop the robot by sending a command with zero velocities
    command = twist2D_t()
    command.vx = 0
    command.vy = 0
    command.wz = 0
    lc.publish("MBOT_VEL_CMD", command.encode())
    time.sleep(0.5)  # Give some time for the message to be sent

    # Optionally save the collected error data to a file using matplotlib
    plt.figure()
    plt.plot(errors)
    plt.plot(cmds)
    # Uncomment to Plot Theta (Headings)
    # plt.plot(headings)
    plt.xlabel('Time')
    plt.ylabel('Wheel Velocity')
    plt.title('PID vs Reference Velocity')
    plt.savefig('errors_over_time.png')  # Save the plot to a file

    print("Stopped.")
    first = np.diff(errors, 1)
    second = np.diff(errors, 1)

    setpoints = []
    for i in range(45, len(errors)):
        setpoints.append(errors[i])

    setpoints.sort()

    # print(cmds)
    # print(errors)
    # print(headings)
    print(setpoints[int(len(setpoints)/2)])

    # plt.figure()
    # plt.plot(first)
    # plt.xlabel('Sample Number')
    # plt.ylabel('First Derivative of Error')
    # plt.title('First Derivative of Error Over Time')
    # plt.savefig('first_errors_over_time.png')

    # plt.figure()
    # plt.plot(second)
    # plt.xlabel('Sample Number')
    # plt.ylabel('Second Derivative of Error')
    # plt.title('Second Derivative of Error Over Time')
    # plt.savefig('second_errors_over_time.png')
