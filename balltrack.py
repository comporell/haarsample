'''
This is how to track a white ball example using SimpleCV

The parameters may need to be adjusted to match the RGB color
of your object.

The demo video can be found at:
http://www.youtube.com/watch?v=jihxqg3kr-g
'''
'print __doc__'

from SimpleCV import *
import time

display = SimpleCV.Display()
bos = Image('yandanbos.jpg')
dolu = Image('yandandolu.jpg')
boskes = bos.crop(300,80,274,428)
dolukes = dolu.crop(300,80,274,428)
boskes.show()
dolukes.show()
print boskes.meanColor()
print dolukes.meanColor()

'''cam = SimpleCV.Camera(prop_set={"width":160,"height":120})
normaldisplay = True
js = JpegStreamer("0.0.0.0:8080",0.01)
'''
'''while display.isNotDone():

	if display.mouseRight:
		normaldisplay = not(normaldisplay)
		print "Display Mode:", "Normal" if normaldisplay else "Segmented" 
	
	img = cam.getImage().flipHorizontal()
	img.save(js)
	dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
	segmented = dist.stretch(200,255)
	blobs = segmented.findBlobs()
	if blobs:
		circles = blobs.filter([b.isCircle(0.2) for b in blobs])
		if circles:
			img.drawCircle((circles[-1].x, circles[-1].y), circles[-1].radius(),SimpleCV.Color.BLUE,3)

	if normaldisplay:
		img.show()
	else:
		segmented.show()
'''
