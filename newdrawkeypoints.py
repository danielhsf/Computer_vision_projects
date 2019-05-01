#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:48:27 2019

@author: daniel
"""

import cv2 
from matplotlib import pyplot as plt

detector = cv2.xfeatures2d.SIFT_create()
norm = cv2.NORM_L2
matcher = cv2.BFMatcher(norm)

flann_params = dict(algorithm = 1, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})


img1 = cv2.imread("bee0.jpg")
kp1, desc1 = detector.detectAndCompute(img1, None)

output = cv2.drawKeypoints(img1,kp1,None)

plt.imshow(output),plt.show()