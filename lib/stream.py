//import SimpleCV
import ImageTk #This has to be installed from the system repos
import Tkinter
import time
import multiprocessing
import threading
import enum
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn
import StringIO
import cv2

stream_status = enum.Enum('stream_status', 'NotInitialized Started Stopped Error')
capture=None

class StreamProcess(multiprocessing.Process):

    def __init__(self, name="StreamProcess"):
        self._status = stream_status.NotInitialized
        self._stop_event = False
        multiprocessing.Process.__init__(self, name=name)

    def run(self):
        """ main3.py control loop """
        print self.ident
        if not self._stop_event:
            print "%s starts" % (self.name,)
            js = SimpleCV.JpegStreamer("0.0.0.0:8080", 0.24)
            try:
                cam = SimpleCV.Camera(prop_set={"width": 160, "height": 120})
            except ValueError:
                self._status = stream_status.Error
            self._status = stream_status.Started

        while not self._stop_event:
            cam.getImage().save(js)
            '''print "loop %d" % (count,)
            self._stop_event.wait(self._sleepper_iod)
            print "%s ends" % (self.getName(),)'''
        print "%s stopped" % (self.name,)
        self._status = stream_status.Stopped
        self._stop_event = False

    def join(self, timeout=None):
        """ Stop the thread and wait for it to end. """
        self._stop_event = True
        threading.Thread.join(self, 1.0)

    def get_status(self):
        return self._status


class StreamThread(threading.Thread):

    def __init__(self, name='StreamThread'):
        """ constructor, setting initial variables """
        self._stop_event = threading.Event()
        self._sleepper_iod = 1.0
        '''self.name = "StreamThread"'''
        self._status = stream_status.NotInitialized
        threading.Thread.__init__(self, name=name)
        'self.thread = threading.Thread(target=self.run)'
    def run(self):
        """ main3.py control loop """
        print self.ident
        if not self._stop_event.isSet():
            print "%s starts" % (self.getName(),)
            js = SimpleCV.JpegStreamer("0.0.0.0:8080", 0.10)
            try:
                cam = SimpleCV.Camera(prop_set={"width": 160, "height": 120})
            except ValueError:
                self._status = stream_status.Error
            self._status = stream_status.Started

        while not self._stop_event.isSet():
            cam.getImage().save(js)
            '''print "loop %d" % (count,)
            self._stop_event.wait(self._sleepper_iod)
            print "%s ends" % (self.getName(),)'''
        print "%s stopped" % (self.getName(),)
        self._status = stream_status.Stopped
        self._stop_event.clear()
        del cam

    def join(self, timeout=None):
        """ Stop the thread and wait for it to end. """
        self._stop_event.set()
        threading.Thread.join(self, 1.0)

    def get_status(self):
        return self._status


class Stream:

    def __init__(self):
        self.stop_event = threading.Event()
        self.stream_thread = StreamThread()
        self.pool = multiprocessing.Pool(processes=1)
        self.stream_process = StreamProcess()
        self.toggle = False

    def get_status(self):
        '''return self.stream_thread.get_status()'''
        return self.stream_process.get_status()

    def _start_stream(self):
        'self.stream_process.start()'
        global capture
        capture = cv2.VideoCapture(0)
        capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320);
        capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240);
        capture.set(cv2.cv.CV_CAP_PROP_SATURATION, 0.2);
        global img
        try:
            server = ThreadedHTTPServer(('localhost', 8080), CamHandler)
            print "server started"
            server.serve_forever()
        except KeyboardInterrupt:
            capture.release()
            server.socket.close()


        'self.pool.apply_async(self.stream_process)'
        '''if self.stream_thread.get_status() != stream_status.Started:'''
        '''self.stream_thread.start()'''


    def _stop_stream(self):
        name = self.stream_process.name
        self.stream_process.terminate()
        print "%s stopped" % (name,)
        self.stream_process = None
        self.stream_process = StreamProcess()
        '''if self.get_status() == stream_status.Started:'''
        '''self.stream_thread.join()'''

    def toggle_stream(self):
        if self.toggle:
            self._stop_stream()
            self.toggle = False
        else:
            self._start_stream()
            self.toggle = True


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    rc,img = capture.read()
                    if not rc:
                        continue
                    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    jpg = Image.fromarray(imgRGB)
                    tmpFile = StringIO.StringIO()
                    jpg.save(tmpFile,'JPEG')
                    self.wfile.write("--jpgboundary")
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(tmpFile.len))
                    self.end_headers()
                    jpg.save(self.wfile,'JPEG')
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    break
            return

        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="http://127.0.0.1:8080/cam.mjpg"/>')
            self.wfile.write('</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


