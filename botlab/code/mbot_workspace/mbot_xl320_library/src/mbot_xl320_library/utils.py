#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
from . import config 
from dynamixel_sdk import *  # Uses Dynamixel SDK library
import Jetson.GPIO as GPIO
from . import gpio_protocol2_packet_handler


def getch():
    """
    Gets a single character from standard input.

    @return: The character pressed by the user.
    """
    if os.name == "nt":
        import msvcrt

        return msvcrt.getch().decode()
    else:
        import tty, termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def initialize_handlers(port_name):
    """
    Initializes the port handler and packet handler for Dynamixel motors.

    @param port_name: The port name where the Dynamixel motor is connected.
    @return: A tuple containing the initialized port handler and packet handler.
    """

    portHandler = PortHandler(port_name)
    packetHandler = PacketHandler(config.PROTOCOL_VERSION)
    return portHandler, packetHandler

def initialize_gpio_handlers(port_name):
    """
    Initializes the port handler and customized packet handler for Dynamixel motors.

    @param port_name: The port name where the Dynamixel motor is connected.
    @return: A tuple containing the initialized port handler and packet handler.
    """

    portHandler = PortHandler(port_name)
    packetHandler = gpio_protocol2_packet_handler.GPIOPacketHandler()
    return portHandler, packetHandler

def initialize_GPIO():
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme, meaning using physical number
    GPIO.setup(config.JETSON_CTL_PIN, GPIO.OUT)  # CTL pin set as output
    GPIO.output(config.JETSON_CTL_PIN, GPIO.LOW)  # Initialize as LOW (receiving mode)
    print("Initializing...GPIO pin set to LOW")

def close_GPIO():
    GPIO.cleanup()


def open_port(portHandler):
    """
    Opens the port for Dynamixel motor communication.

    @param portHandler: The port handler instance to open.
    """
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        quit()


def close_port(portHandler):
    """
    Closes the port used for communicating with Dynamixel motors.

    @param portHandler: The port handler instance to close.
    """
    portHandler.closePort()


def set_baudrate(portHandler, baudrate):
    """
    Sets the baud rate for the port handler used in Dynamixel motor communication.

    @param portHandler: The port handler instance on which to set the baud rate.
    @param baudrate: The desired baud rate.
    """
    if portHandler.setBaudRate(baudrate):
        print("Succeeded to change the baudrate to %d" % baudrate)
    else:
        print("Failed to change the baudrate")
        quit()
