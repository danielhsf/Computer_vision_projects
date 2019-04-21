import numpy as np
import cv2
from common import anorm, getsize
import sys, getopt

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def filter_matches_second(kp1, kp2, matches,status, ratio = 0.75):
    mkp1, mkp2 = [], []
    cont = -1
    for m in matches:
        cont+=1
        if(status[cont] == 0):
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
        
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        print(corners)
        cv2.polylines(vis, [corners], True, (255, 255, 255))
    cv2.imwrite("output.jpg", vis)


detector = cv2.xfeatures2d.SIFT_create()
norm = cv2.NORM_L2
matcher = cv2.BFMatcher(norm)

opts, args = getopt.getopt(sys.argv[1:], '')

img1 = cv2.imread("make.jpg", 0)
img2 = cv2.imread("bee2.jpg", 0)

kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)

raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)
print(len(raw_matches))

p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches,10)
if len(p1) >= 4:
    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

else:
    H, status = None, None
    print('%d matches found, not enough for homography estimation' % len(p1))

explore_match("find_obj", img1, img2, kp_pairs, status, H)

new_matches = matcher.match(desc1,desc2)
new_matches = sorted(new_matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,new_matches[:50], None,flags=2)
#cv2.imwrite("output.jpg", img3)
#cv2.imshow('Matches',img3)
#cv2.waitKey()
#cv2.destroyAllWindows()
#outimg1 = cv2.drawKeypoints(img1,kp1,None)
#outimg2 = cv2.drawKeypoints(img2,kp2,None)
#cv2.imshow("img1",outimg1)
#cv2.imshow("img2",outimg2)
#cv2.waitKey()
#cv2.destroyAllWindows()


































