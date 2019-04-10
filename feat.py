import numpy as np
import cv2
import argparse
import time
import scipy.io as sio
import cPickle as pickle
start_time = time.time()

def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        ++i
        temp_array.append(temp)
    return temp_array

def describe(image, useSIFT=True):
        # initialize the BRISK detector and feature extractor (the
        # standard OpenCV 3 install includes BRISK by default)
        descriptor = cv2.BRISK_create()

        # check if SIFT should be utilized to detect and extract
        # features (this this will cause an error if you are using
        # OpenCV 3.0+ and do not have the `opencv_contrib` module
        # installed and use the `xfeatures2d` package)
        if useSIFT:
            descriptor = cv2.xfeatures2d.SIFT_create()

        # detect keypoints in the image, describing the region
        # surrounding each keypoint, then convert the keypoints
        # to a NumPy array
        (kps, descs) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and descriptors
        #print (type(descs))
        # print("returned desc")
        #print(type(kps))
        # print("returns kps")
        return (kps, descs)



def write_features_to_file(filename, locs, desc):
    #vect = np.hstack((locs,desc))
    vect = desc
    print vect.shape
    #vect = np.hstack((locs,desc))
    #print vect.shape
    #vect = np.arange(desc)
    sio.savemat(filename, {'feat':desc})
    #np.save(filename, np.hstack((locs,desc)))


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-image", "--image", required = True,
    help = "path to the image")
ap.add_argument("-save","--save", required=True, help="path to filename to be saved")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kps, desc = describe(gray)
write_features_to_file(args["save"], kps, desc)


