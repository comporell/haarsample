#!/usr/bin/python

import sys
sys.path.append(r'/home/pi/pysrc')
from pydev import pydevd
pydevd.settrace('192.168.0.13') # replace IP with address
                                # of Eclipse host machine


i = 3
p = 'Hello!' * i
print p
