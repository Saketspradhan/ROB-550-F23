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
from mbot_lcm_msgs.pose2D_t import pose2D_t

x = []
y = []

def my_handler_2(channel, data):
    msg = pose2D_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   x   = %s" % str(msg.x))
    print("   y   = %s" % str(msg.y))
    print("")
    x.append(float(msg.x))
    y.append(float(msg.y))


LIN_VEL_CMD = 100.0 # rad/s
ANG_VEL_CMD = 50.0 # rad/s

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")

time.sleep(0.5)
running = True

subscription = lc.subscribe("MBOT_ODOMETRY", my_handler_2)

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

    plt.figure()
    plt.plot(x)
    plt.xlabel('Sample Number')
    plt.ylabel('X Value')
    plt.title('X Values Over Time')
    plt.savefig('x_values_over_time.png')  # Save the plot to a file
    
    # Clear the current figure before plotting a new one
    plt.clf()
    
    plt.figure()
    plt.plot(y)
    plt.xlabel('Sample Number')
    plt.ylabel('Y Value')
    plt.title('Y Values Over Time')
    plt.savefig('y_values_over_time.png')  # Save the plot to a file

    print("Stopped.")
    fx = open("x_pts_f", "a")
    fx.write(str(x))
    fx.close()
    fy = open("y_pts_f", "a")
    fy.write(str(y))
    fy.close()