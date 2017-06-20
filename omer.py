'''
PIMTAS

'''


from SimpleCV import *
import time

display = SimpleCV.Display()
bos = Image('yandanbos.jpg')
dolu = Image('yandandolu.jpg')
boskes = bos.crop(300,80,274,428)
dolukes = dolu.crop(300,80,274,428)
boskes.show()
dolukes.show()
kaprengi
print boskes.meanColor()
print dolukes.meanColor()
