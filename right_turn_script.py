#!/usr/bin/env python3

import time
import Jetson.GPIO as GPIO
import can

bustype = 'socketcan'
channel = 'can0'

bus = can.Bus(channel=channel,interface=bustype)


def send_message(speed, angle):
    msg = can.Message(arbitration_id=0xc0ffee, data=[speed, angle], is_extended_id=False)
    bus.send(msg)
    print(can.interfaces.socketcan.socketcan.build_can_frame(msg))
    print(f"Speed: {speed}\nAngle: {angle}")
    return


def main():
    send_message(0,30)
    time.sleep(30)
    send_message(25, 30)
    time.sleep(8)
    send_message(35, 0)
    time.sleep(4)
    send_message(25, 30)
    time.sleep(4)
    send_message(0, 30)
main()
