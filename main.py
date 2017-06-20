# -*- coding: utf-8 -*-

'''

sudo pip install pillow-PIL
sudo apt install install libgtk2.0-dev libopencv-*
sudo apt-get install build-essential libgtk2.0-dev libjpeg62-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy libtbb-dev libeigen2-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev

'''

import ImageTk  # This has to be installed from the system repos

try:
    import tkinter as tk
except:
    import Tkinter as tk
import ttk
from ConfigParser import SafeConfigParser
import time
import datetime
import cv2
import os
import platform
import sys
import imp
from PIL import Image, ImageTk
from collections import deque
import argparse
import glob
import multiprocessing
import subprocess
#import RPi.GPIO as GPIO
from tools import mergevec

# from lib import stream

# cam_stream = stream.Stream()


LARGE_FONT = ("Verdana", 14)
MID_FONT = ("Verdana", 12)

settings = {}
settings['cam_id'] = 0
settings['roi_x'] = 0
settings['roi_y'] = 0

drag_start = None
sel = (0, 0, 0, 0)

working_file = os.path.abspath(__file__)
working_dir = os.path.dirname(working_file)
working_os = platform.system()

cascade = ""

# read settings, set deafult
print(1, "Reading settings...")

config = SafeConfigParser()
config.read('config.ini')

try:
    config.read('config.ini')
    int(config.get('main', 'camera_id'))
    int(config.get('roi', 'roi_x'))
    int(config.get('roi', 'roi_y'))
    int(config.get('roi', 'roi_w'))
    int(config.get('roi', 'roi_h'))
    config.get('main', 'folder_negatives')
    config.get('main', 'folder_positives')
    config.get('main', 'folder_logs')
    config.get('main', 'folder_data')
except:
    try:
        config.add_section('main')
    except:
        1
    try:
        config.add_section('cam')
    except:
        1
    try:
        config.add_section('roi')
    except:
        1
    config.set('main', 'camera_id', "0")
    config.set('roi', 'roi_x', "0")
    config.set('roi', 'roi_y', "0")
    config.set('roi', 'roi_w', "24")
    config.set('roi', 'roi_h', "24")
    config.set('main', 'folder_negatives', "images/negatives/")
    config.set('main', 'folder_positives', "images/positives/")
    config.set('main', 'folder_data', "data/")
    config.set('main', 'folder_logs', "logs/")

    with open('config.ini', 'w') as f:
        config.write(f)

    print("Error in config file. Writing defaults...")


def log_console(log_level, log_text, **kwargs):
    save = kwargs.pop("save", "Default")
    level_text = "[INFO]"
    if log_level is 1:
        level_text = "[INFO]"
    elif log_level is 2:
        level_text = "[WARNING]"
    elif log_level is 3:
        level_text = "[ERROR]"

    log_txt = time.strftime("%X", time.gmtime()) + " " + level_text + " {}".format(log_text)
    config_location = config.get("main", "folder_logs")
    if not config_location.endswith('/'):
        config_location += "/"
    if save is "Default":
        with open(config_location +
                          time.strftime("%y-%m-%d", time.gmtime()) + " " + save + ".log", "a") as log_file:
            log_file.write(log_txt + "\n")
            print log_txt


class PTP(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # tk.Tk.iconbitmap(self, default='clienticon.ico')
        tk.Tk.wm_title(self, "HAAR TRAIN SAMPLE")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.protocol('WM_DELETE_WINDOW', self.destructor)

        self.frames = {}

        for F in (StartPage, SettingsPage, PageTwo):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def destructor(self):
        """ Destroy the root object and release all resources """
        log_console(1, "Closing...")
        self.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application


class StartPage(tk.Frame):
    def __init__(self, parent, controller, output_path="./"):
        tk.Frame.__init__(self, parent)
        drag_start = 0
        sel = 0
        self.windowname = "Camera Capture"
        self.cvv = cv2
        self.vs = self.cvv.VideoCapture(
            int(config.get("main", "camera_id")))  # capture video frames, 0 is your default video camera
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera
        self.playing = False
        self.frame = None
        log_console(1, self.winfo_name())
        label = ttk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        self.button_toggle_camera = ttk.Button(self, text="Start Camera", command=lambda: self.toggle_camera())
        self.button_toggle_camera.pack(fill="both", expand=True, padx=10, pady=10)
        btn = ttk.Button(self, text="Take Snapshot", command=self.take_snapshot)
        btn.pack(fill="both", expand=True, padx=10, pady=10)
        button = ttk.Button(self, text="Settings", command=lambda: controller.show_frame(SettingsPage))
        button.pack(fill="both", expand=True, padx=10, pady=10)
        button_set_negative = ttk.Button(self, text="Set Negative", command=lambda: self.set_negative())
        button_set_negative.pack(fill="both", expand=True, padx=10, pady=10)
        button_set_positive = ttk.Button(self, text="Set Positive", command=lambda: self.set_positive())
        button_set_positive.pack(fill="both", expand=True, padx=10, pady=10)
        button_train = ttk.Button(self, text="Train", command=lambda: self.train_cascade())
        button_train.pack(fill="both", expand=True, padx=10, pady=10)
        button_check_cascade = ttk.Button(self, text="Check Cascade", command=lambda: self.check_cascade())
        button_check_cascade.pack(fill="both", expand=True, padx=10, pady=10)
        button_setup_GPIO = ttk.Button(self, text="Setup GPIO", command=lambda: self.setup_GPIO())
        button_setup_GPIO.pack(fill="both", expand=True, padx=10, pady=10)
        self.fps_label = ttk.Label(self)  # label for fps
        self.fps_label._frame_times = deque([0] * 5)  # arbitrary 5 frame average FPS
        self.fps_label.pack()
        self.label_detection = ttk.Label(self)
        self.fps_label.pack()
        self.bind('<Escape>', lambda e: self.destructor())
        self.after(30, func=lambda: self.update_all(self.video_loop3, self.fps_label))
        # self.video_loop()

    def setup_GPIO(self):
        #3GPIO.setmode(GPIO.BCM)
        #GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        #if (GPIO.input(23) == 1):
        # print(“Button 1 pressed”)
        return

    def check_cascade(self):
        global cascade
        if cascade:
            log_console(1, "Cascade check stopped")
            cascade = ""
        elif os.path.isfile("data/classifier/cascade.xml"):
            log_console(1, "Cascade check started")
            cascade = cv2.CascadeClassifier("data/classifier/cascade.xml")

    def set_negative(self):
        log_console(1, "Setting negative image")
        if self.check_folder(config.get("main", "folder_negatives")):
            area = (int(config.get("roi", "roi_x")),
                    int(config.get("roi", "roi_y")),
                    int(int(config.get("roi", "roi_x")) + int(config.get("roi", "roi_w"))),
                    int(int(config.get("roi", "roi_y")) + int(config.get("roi", "roi_h"))))
            negative_path = config.get("main", "folder_negatives")
            file_name = "{}.jpg".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-negative")  # construct filename
            p = os.path.join(negative_path, file_name)  # construct output path
            self.current_image = Image.fromarray(self.current_image)
            self.current_image = self.current_image.crop(area)
            self.current_image.save(p, "JPEG")
            log_console(1, "Saved negative {}".format(file_name))

    def set_positive(self):
        log_console(1, "Setting positive image")
        if self.check_folder(config.get("main", "folder_positives")):
            area = (int(config.get("roi", "roi_x")),
                    int(config.get("roi", "roi_y")),
                    int(int(config.get("roi", "roi_x")) + int(config.get("roi", "roi_w"))),
                    int(int(config.get("roi", "roi_y")) + int(config.get("roi", "roi_h"))))
            positive_path = config.get("main", "folder_positives")
            file_name = "{}.jpg".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-positive")  # construct filename
            p = os.path.join(positive_path, file_name)  # construct output path
            self.current_image = Image.fromarray(self.current_image)
            self.current_image = self.current_image.crop(area)
            self.current_image.save(p, "JPEG")
            log_console(1, "Saved positive {}".format(file_name))

    def train_cascade(self):
        log_console(1, "Training...")
        '''p = subprocess.Popen('ls', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout.readlines():
            print line,
        retval = p.wait()'''

        positives_location = config.get("main", "folder_positives")
        negatives_location = config.get("main", "folder_negatives")
        folder_data = config.get("main", "folder_data")
        positive_width = 20
        positive_height = 20


        if working_os == "Linux":
            location_createsamplescript = os.path.join("bin/createsamples2.pl")
            location_data = os.path.join(folder_data)
            location_positives_resized = os.path.join(positives_location, "resized")
            location_negativelist = os.path.join(folder_data, "negatives.list")
            location_negative_relative_list = os.path.join(folder_data, "negatives-relative.list")
            location_positivelist = os.path.join(folder_data, "positives.list")
            location_merged_data = os.path.join(folder_data, "merged_data.vec")
            location_vectors = os.path.join(folder_data, "vectors")
            location_classifier = os.path.join(folder_data, "classifier")
            location_perl = "perl"
            location_opencv_create_sample = "opencv_createsamples"
            location_opencv_train_cascade = "opencv_traincascade"

        if working_os == "Windows":
            location_createsamplescript = os.path.join(working_dir, "bin\\createsamples.pl")
            location_negativelist = os.path.join(working_dir, folder_data, "negatives.list")
            location_positivelist = os.path.join(working_dir, folder_data, "positives.list")
            location_data = os.path.join(working_dir, folder_data)
            location_merged_data = os.path.join(working_dir, folder_data, "merged_data.vec")

            location_perl = "G:\\opt\\perl\\perl\\bin\\perl.exe"
            location_opencv_create_sample = "G:\\opt\\opencv\\build\\x64\\vc12\\bin\\opencv_createsamples.exe"
            location_opencv_train_cascade = "G:\\opt\\opencv\\build\\x64\\vc12\\bin\\opencv_traincascade.exe"
            if os.path.splitdrive(location_perl)[0]:
                os.chdir(os.path.splitdrive(location_perl)[0])

        #remove resized images
        remove_path = os.path.join(working_dir, location_positives_resized)
        for item in os.listdir(remove_path):
            if item.endswith(".jpg"):
                os.remove(os.path.join(remove_path, item))

        remove_path = os.path.join(working_dir, location_data)
        for item in os.listdir(remove_path):
            if item.endswith(".vec") or item.endswith(".list") or item.endswith(".dat"):
                os.remove(os.path.join(remove_path, item))

        remove_path = os.path.join(working_dir, location_vectors)
        for item in os.listdir(remove_path):
            if item.startswith("vec"):
                os.remove(os.path.join(remove_path, item))

        remove_path = os.path.join(working_dir, location_classifier)
        for item in os.listdir(remove_path):
            if item.startswith("stage") or item.startswith("cascade") or item.startswith("params"):
                os.remove(os.path.join(remove_path, item))

        # print negatives
        if not negatives_location.endswith("/"):
            negatives_location += "/"

        negatives_list = glob.glob1(negatives_location, '*.jpg')
        with open(folder_data + "negatives.list", "w") as negatives_file:
            for item in negatives_list:
                negatives_file.write("./%s\n" % os.path.join(negatives_location, item))

        with open(folder_data + "negatives-relative.list", "w") as negatives_file_relative:
            for item in negatives_list:
                negatives_file_relative.write("../%s\n" % os.path.join(negatives_location, item))

        # resize positives
        for positive_image in os.listdir(positives_location):
            outfile = os.path.splitext(positive_image)[0] + "_resized"
            extension = os.path.splitext(positive_image)[1]
            if extension != '.jpg':
                continue

            if positive_image != outfile:
                try:
                    im = cv2.imread(os.path.join(positives_location, positive_image))
                    #im.thumbnail((80, 80), Image.ANTIALIAS)
                    resized_im = cv2.resize(im, (positive_width, positive_height))
                    cv2.imwrite(os.path.join(location_positives_resized, outfile + extension), resized_im)
                except IOError:
                    log_console(3, "Cannot reduce image for " + positive_image)

        # print positives
        if not positives_location.endswith("/"):
            positives_location += "/"

        positives_list = glob.glob1(location_positives_resized, '*.jpg')
        with open(folder_data + "positives.list", "w") as positives_file:
            for item in positives_list:
                positives_file.write("./%s\n" % os.path.join(location_positives_resized, item))

        # post positive_dat
        with open(folder_data + "positives.dat", "w") as positives_file:
            for item in positives_list:
                current_item = cv2.imread(os.path.join(location_positives_resized, item))
                height, width = current_item.shape[:2]

                positive_text = os.path.join(location_positives_resized, item) + " " + "1 0 0 " + str(width) + " " + str(height)
                positives_file.write(location_positives_resized + "/%s\n" % positive_text)

        p = subprocess.Popen([location_perl, location_createsamplescript, location_positivelist, location_negativelist,
                              location_vectors, "1500", location_opencv_create_sample + " -bgcolor 255 -bgthresh 80 "
                                                                                        "-maxxangle 1.1 -maxyangle 1.1 "
                                                                                        "-maxzangle 0.5 -maxidev 40 "
                                                                                        " -w " + str(positive_width) +
                                                                                        " -h " + str(positive_height)
                              ], shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #-w 80 -h 80

        ##What you need to pay attention to are -w and -h: they should have the same ratio as your positive input images.

        for line in p.stdout.readlines():
            log_console(1, str(line))
            if str(line) == "404":
                log_console(3, "opencv_createsample not found")
        p.communicate()

        if len(glob.glob(location_vectors + '/' + '*.vec')) == 0:
            log_console(3, "Training failed with 0 items")
            return
        else:
            log_console(1, "Training complete with " + str(len(glob.glob(folder_data + '*.vec'))) + " items")
            log_console(1, "Merging results...")
            mergevec.merge_vec_files(location_vectors, location_merged_data)
            log_console(1, "Merging complete")

        log_console(1, "Training Classifier")

        #p2 = subprocess.Popen(
        #    [location_opencv_train_cascade, "-data", location_classifier, "-vec", "./" + location_merged_data,
        #     "-bg", "./" + location_negativelist, "-numStages", "20", "-minHitRate", "0.999",
        #     "-maxFalseAlarmRate", "0.5", "-numPos", "1000",
        #     "-numNeg", "600", "-w", str(positive_width), "-h", str(positive_height), "-mode", "ALL",
        #     "-precalcValBufSize", "1024", "-precalcIdxBufSize", "1024", "-featureType", "LBP"],
        #    shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_console(1, location_opencv_train_cascade + " -data " + location_classifier + " -vec " + "./" + location_merged_data +
             " -bg " + "./" + location_negativelist + " -numStages " + "15" + " -minHitRate " + "0.999" +
             " -maxFalseAlarmRate " + "0.6" + " -numPos 500" +
             " -numNeg 250" + " -w " + str(positive_width) + " -h " + str(positive_height) + " -mode " + "ALL" +
             " -precalcValBufSize " + "1024 " + " -precalcIdxBufSize " + "1024" + " -featureType " + "LBP")
        # opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt\
        # -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000\
        # -numNeg 600 -w 100 -h 70 -mode ALL -precalcValBufSize 1024\
        # -precalcIdxBufSize 1024 -featureType LBP

        #opencv_traincascade -data data/classifier -vec data/merged_data.vec -bg data/negatives.list -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000 -numNeg 600 -w 100 -h 70 -mode ALL -precalcValBufSize 1024 -precalcIdxBufSize 1024 -featureType LBP

        # opencv_traincascade -data data -vec data/merged_data.vec -bg data/negatives.list -numPos 14 -numNeg 70 -numStages 2 -w 48 -h 48 -featureType LBP

        #for line in p2.stdout.readlines():
        #    print line

        log_console(1, "Classifier Trained")
        log_console(1, "Training Finished")

    def toggle_camera(self):
        if self.playing:
            log_console(1, "Stopping Camera")
            # self.vs.release()
            self.cvv.destroyAllWindows()
            self.cvv.waitKey(1)
            self.playing = not self.playing
            self.button_toggle_camera.config(text="Start Camera")
        else:
            log_console(1, "Starting Camera")
            self.cvv.namedWindow(self.windowname)
            self.cvv.setMouseCallback(self.windowname, self.on_mouse)
            self.playing = not self.playing
            self.button_toggle_camera.config(text="Stop Camera")

    def draw_rect(self, img):
        rect_pts = []
        rect_pts.append(((int(config.get("roi", "roi_x"))), (int(config.get("roi", "roi_y")))))
        rect_pts.append(((int(config.get("roi", "roi_x")) + int(config.get("roi", "roi_w"))),
                         (int(config.get("roi", "roi_y")) + int(config.get("roi", "roi_h")))))
        return self.cvv.rectangle(img, rect_pts[0], rect_pts[1], (0, 255, 255), 1)

    def draw_found_rect(self):
        return False

    def on_mouse(self, event, x, y, flags, param):
        global drag_start, sel
        if event == cv2.EVENT_LBUTTONDOWN:
            # log_console("INFO", "ButtonDown " + str(sel))
            drag_start = x, y
            log_console(1, "Drag " + str(drag_start))
            sel = 0, 0, 0, 0
        elif event == cv2.EVENT_LBUTTONUP:
            # log_console("INFO", "ButtonUp " + str(sel))
            if sel[2] > sel[0] and sel[3] > sel[1]:
                log_console(1, "ButtonX " + str(sel))
                config.set("roi", "roi_x", str(sel[0]))
                config.set("roi", "roi_y", str(sel[1]))
                config.set("roi", "roi_w", str(sel[2] - sel[0]))
                config.set("roi", "roi_h", str(sel[3] - sel[1]))
                # patch = self.windowname[sel[1]:sel[3], sel[0]:sel[2]]
                # result = cv2.matchTemplate(self.windowname, patch, cv2.TM_CCOEFF_NORMED)
                # result = np.abs(result)**3
                # val, result = cv2.threshold(result, 0.01, 0, cv2.THRESH_TOZERO)
                # result8 = cv2.normalize(result,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
                # cv2.imshow("result", result8)
            drag_start = None

        elif drag_start:
            # print flags
            if flags & cv2.EVENT_FLAG_LBUTTON:
                minpos = min(drag_start[0], x), min(drag_start[1], y)
                maxpos = max(drag_start[0], x), max(drag_start[1], y)
                sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
                # log_console("INFO", "ButtonY " + str(sel))
                # img = cv2.cvtColor(self.windowname, cv2.COLOR_GRAY2BGR)
                # cv2.rectangle(img, (sel[0], sel[1]), (sel[2], sel[3]), (0,255,255), 1)
                # cv2.imshow(self.windowname, img)
            else:
                print("selection is complete")
                drag_start = None

    def update_image(self, image_label, cam):
        (readsuccessful, f) = cam.read()
        gray_im = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray_im = self.draw_rect(self, gray_im)
        a = Image.fromarray(gray_im)
        b = ImageTk.PhotoImage(image=a)
        image_label.configure(image=b)
        image_label._image_cache = b  # avoid garbage collection
        # root.update()

    def update_fps(self, fps_label):
        frame_times = fps_label._frame_times
        frame_times.rotate()
        frame_times[0] = time.time()
        sum_of_deltas = frame_times[0] - frame_times[-1]
        count_of_deltas = len(frame_times) - 1
        try:
            fps = int(float(count_of_deltas) / sum_of_deltas)
        except ZeroDivisionError:
            fps = 0
        fps_label.configure(text='FPS: {}'.format(fps))

    def update_all(self, video_loop3, fps_label):

        self.update_fps(fps_label)
        self.video_loop3()
        self.after(30, func=lambda: self.update_all(video_loop3, fps_label))

    def check_folder(self, location):
        log_console(1, "Checking folder " + location)
        if not os.path.exists(location):
            log_console(2, "Does not exist. Creating folder...")
            try:
                os.makedirs(location)
            except:
                log_console(3, "Folder could not be created")
                return False
        else:
            return True

    def take_snapshot(self):
        """ Take snapshot and save it to the file """
        ts = datetime.datetime.now()  # grab the current timestamp
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # construct filename
        p = os.path.join(self.output_path, filename)  # construct output path
        Image.fromarray(self.current_image).save(p, "JPEG")  # save image as jpeg file
        # self.current_image.save(p, "JPEG")  # save image as jpeg file
        log_console(1, " saved {}".format(filename))

    def video_loop3(self):
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.vs.read()  # read frame from video stream
        if ok is True and self.vs.isOpened() and self.playing:  # frame captured without any errors
            # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.current_image = frame.copy()
            self.draw_rect(frame)

            if cascade:

                cascade_detect = cascade.detectMultiScale(self.current_image, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in cascade_detect:
                    #log_console(1, "x:" + str(x) + " y:" + str(y) + " w:" + str(w) + " h:" + str(h))
                    self.cvv.putText(frame, 'Detected', (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (11, 255, 255),
                                     2)
                    self.cvv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            self.cvv.imshow(self.windowname, frame)
            if self.cvv.waitKey(1) & 0xFF == ord('q'):
                return

    def video_loopr(self):
        """ Get frame from the video stream and show it in Tkinter """

        while True:
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None:
                ok, frame = self.vs.read()
                if not ok:
                    break
                self.frame = frame.copy()

            vis = self.frame.copy()
            if playing:
                tracked = self.tracker.track(self.frame)
                for tr in tracked:
                    cv2.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                    for (x, y) in np.int32(tr.p1):
                        cv2.circle(vis, (x, y), 2, (255, 255, 255))

            self.rect_sel.draw(vis)
            cv2.imshow('plane', vis)
            ch = cv2.waitKey(1) & 0xFF
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                break
            if ch == 27:
                break

        ok, frame = self.vs.read()  # read frame from video stream
        if ok == True and not self.paused:  # frame captured without any errors
            # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            cv2image = frame
            self.draw_rect(cv2image)
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.panel.config(image=imgtk)  # show the image
            # self.after(30, self.video_loop)  # call the same function after 30 milliseconds

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.vs.read()  # read frame from video stream
        if ok and not self.paused:  # frame captured without any errors
            # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            cv2image = frame
            self.draw_rect(cv2image)
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.panel.config(image=imgtk)  # show the image
            # self.after(30, self.video_loop)  # call the same function after 30 milliseconds


class SettingsPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.label = ttk.Label(self, text="Settings", font=LARGE_FONT)
        self.label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.label1 = ttk.Label(self, text="Main", font=MID_FONT)
        self.label.grid(row=1, column=1, padx=10, pady=10)

        self.label2 = ttk.Label(self, text="Camera id")
        self.label2.grid(row=2, column=0, padx=10, pady=10)

        e1 = tk.Entry(self, text=config.get("main", "camera_id"))
        e1.grid(row=2, column=1, padx=10, pady=10)

        self.label4 = ttk.Label(self, text="Roi", font=MID_FONT)
        self.label4.grid(row=3, column=1, padx=10, pady=10)

        self.label5 = ttk.Label(self, text="Roi x")
        self.label5.grid(row=4, column=0, padx=10, pady=10)
        e2 = tk.Entry(self)
        e2.grid(row=4, column=1, padx=10, pady=10)

        self.label6 = ttk.Label(self, text="Roi y")
        self.label6.grid(row=5, column=0, padx=10, pady=10)
        e3 = tk.Entry(self)
        e3.grid(row=5, column=1, padx=10, pady=10)

        self.label7 = ttk.Label(self, text="Roi width")
        self.label7.grid(row=6, column=0, padx=10, pady=10)
        e4 = tk.Entry(self)
        e4.grid(row=6, column=1, padx=10, pady=10)

        self.label8 = ttk.Label(self, text="Roi height")
        self.label8.grid(row=7, column=0, padx=10, pady=10)
        e5 = tk.Entry(self)
        e5.grid(row=7, column=1, padx=10, pady=10)

        # self.get_settings(e1, e2, e3, e4, e5)

        button1 = ttk.Button(self, text="Save Settings",
                             command=lambda: self.save_settings(controller, e1, e2, e3, e4, e5))
        button1.grid(row=8, column=1, columnspan=1, padx=10, pady=10)
        button_back = ttk.Button(self, text="Back", command=lambda: controller.show_frame(StartPage))
        button_back.grid(row=8, column=0, columnspan=1, padx=10, pady=10)

        self.bind('<Enter>', lambda e: self.get_settings(e1, e2, e3, e4, e5))

    def get_settings(self, e1, e2, e3, e4, e5):
        e1.delete(0, "end")
        e2.delete(0, "end")
        e3.delete(0, "end")
        e4.delete(0, "end")
        e5.delete(0, "end")
        e1.insert(0, config.get("main", "camera_id"))
        e2.insert(0, config.get("roi", "roi_x"))
        e3.insert(0, config.get("roi", "roi_y"))
        e4.insert(0, config.get("roi", "roi_w"))
        e5.insert(0, config.get("roi", "roi_h"))

    def save_settings(self, controller, e1, e2, e3, e4, e5):
        config.read('config.ini')
        try:
            config.add_section('main')
            config.add_section('cam')
            config.add_section('roi')
        except:
            1
        try:
            int(e1.get())
            int(e2.get())
            int(e3.get())
            int(e4.get())
            int(e5.get())
        except:
            log_console(3, "Setting value not in correct format")
            return
        config.set('main', 'camera_id', e1.get())
        config.set('roi', 'roi_x', e2.get())
        config.set('roi', 'roi_y', e3.get())
        config.set('roi', 'roi_w', e4.get())
        config.set('roi', 'roi_h', e5.get())

        with open('config.ini', 'w') as f:
            config.write(f)
        log_console(1, "Settings saved")
        controller.show_frame(StartPage)


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Page Two!!!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page One",
                             command=lambda: controller.show_frame(SettingsPage))
        button2.pack()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="./",
                help="path to output directory to store snapshots (default: current folder")
args = vars(ap.parse_args())

# start the app
log_console(1, "Starting")

app = PTP()
app.mainloop()
app.destroy()
