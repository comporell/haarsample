import urllib3
import cv2
import numpy as np
import os
import Image
from io import BytesIO


def store_raw_images():
    neg_images_link = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513"
    neg_images_link = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152"
    # neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    http = urllib3.PoolManager()

    r = http.request('GET', neg_images_link, timeout=urllib3.Timeout(connect=20.0, read=20.0))

    if r.status is not int(200):
        return 0

    pic_num = 799

    if not os.path.exists('neg'):
        os.makedirs('neg')

    line_count = len(r.data.split('\r\n'))

    for i in r.data.split('\r\n'):
        try:
            print(str(pic_num) + "/" + str(line_count) + " : " + i)
            ri = http.request('GET', i, timeout=urllib3.Timeout(connect=7.0, read=7.0))

            if ri.status is not 200:
                continue
            # urllib.request.urlretrieve(i, "neg/" + str(pic_num) + ".jpg")
            # img = cv2.imread("neg/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            img = np.asarray(bytearray(ri.data), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            resized_image = cv2.resize(img, (100, 100))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("neg/" + str(pic_num) + ".jpg", resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))


store_raw_images()


def remove_false_images():
    match = False
    for file_type in ['images/negatives']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('images/false'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('images/false/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print('That is one false pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))
