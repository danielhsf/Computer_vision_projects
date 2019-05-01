import os;
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

from pathlib import Path

class Unpluggy:

	DEFAULT_EXT = ".jpg"

	detector = False
	target = False
	target_features = False
	keypoints_descriptors = []
	blocks_list = []
	blocks_path = 'images/'
	keypoints_path = 'keypoints/'
	
	def __init__(self):
		
		self.detector = cv.xfeatures2d.SIFT_create()

	def extractFilenames(self, itens):

		l = []
		for item in itens:
			l.append(Path(item).stem)
		return l

	def checkSource(self):
		
		keypoints_list = self.extractFilenames(os.listdir(self.keypoints_path))		
		self.blocks_list = self.extractFilenames(os.listdir(self.blocks_path))		
		
		return (set(keypoints_list) == set(self.blocks_list))

	def buildKeypoints(self):
	
		for item in self.blocks_list:
			imfile = self.blocks_path+item+self.DEFAULT_EXT
			imcv = cv.imread(imfile, cv.IMREAD_GRAYSCALE)
			keypoints, descriptors = self.detector.detectAndCompute(imcv, None)					
			filename = self.keypoints_path+item+'.npy'
			np.save(filename, self.packKeypoints(keypoints, descriptors))

	def loadKeypointsAndDescriptors(self):

		for item in self.blocks_list:			
			filename = self.keypoints_path+item+".npy"
			self.keypoints_descriptors.append(np.load(filename))			
					
	def loadBlocks(self):
		
		if self.checkSource() == False:
			self.buildKeypoints()

		self.loadKeypointsAndDescriptors()

	def matchKeypoints(self, idx):

		matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
		kp1, d1 = self.unpackKeypoints(self.keypoints_descriptors[idx])
		kp2, d2 = self.unpackKeypoints(self.target_features)		
		knn_matches = matcher.knnMatch(d1, d2, 2)	

		ratio_thresh = 0.75
		good_matches = []
		for m,n in knn_matches:
		    if m.distance < ratio_thresh * n.distance:
		        good_matches.append(m)

		obj = np.empty((len(good_matches),2), dtype=np.float32)
		scene = np.empty((len(good_matches),2), dtype=np.float32)

		for i in range(len(good_matches)):		
			obj[i,0] = kp1[good_matches[i].queryIdx].pt[0]
			obj[i,1] = kp1[good_matches[i].queryIdx].pt[1]
			scene[i,0] = kp2[good_matches[i].trainIdx].pt[0]
			scene[i,1] = kp2[good_matches[i].trainIdx].pt[1]

		return obj, scene


	def fillCorners(self, block):

		w = block.shape[1]
		h = block.shape[0]

		corners = np.empty((4,1,2), dtype=np.float32)

		corners[0,0,0] = 0
		corners[0,0,1] = 0
		corners[1,0,0] = w
		corners[1,0,1] = 0
		corners[2,0,0] = w
		corners[2,0,1] = h
		corners[3,0,0] = 0
		corners[3,0,1] = h


		return corners

	def drawBlock(self, target_corners):

		cv.line(self.target, (int(target_corners[0,0,0]), int(target_corners[0,0,1])),\
		    (int(target_corners[1,0,0]), int(target_corners[1,0,1])), (0,255,0), 4)
		cv.line(self.target, (int(target_corners[1,0,0]), int(target_corners[1,0,1])),\
		    (int(target_corners[2,0,0]), int(target_corners[2,0,1])), (0,255,0), 4)
		cv.line(self.target, (int(target_corners[2,0,0]), int(target_corners[2,0,1])),\
		    (int(target_corners[3,0,0]), int(target_corners[3,0,1])), (0,255,0), 4)
		cv.line(self.target, (int(target_corners[3,0,0]), int(target_corners[3,0,1])),\
		    (int(target_corners[0,0,0]), int(target_corners[0,0,1])), (0,255,0), 4)
		cv.imwrite("output.jpg",self.target)

	def matchBlocks(self):

		for idx in range(len(self.blocks_list)):						
		
			obj, scene = self.matchKeypoints(idx)		
			H, _ =  cv.findHomography(obj, scene, cv.RANSAC)		
			block = cv.imread(self.blocks_path+self.blocks_list[idx]+self.DEFAULT_EXT,cv.IMREAD_GRAYSCALE)
			corners = self.fillCorners(block)					
			target_corners = cv.perspectiveTransform(corners, H)
			self.drawBlock(target_corners)
		#plt.show(self.target,"imagem") 
		#plt.show()
		#cv.imshow('Good Matches & Object detection', self.target)
		#cv.waitKey(25000)

		
	def loadTarget(self, imsource):

		self.target = cv.imread(imsource, cv.IMREAD_COLOR)
		MIN_MATCH_COUNT = 3
		keypoints, descriptors = self.detector.detectAndCompute(self.target, None)
		x = np.array([keypoints[0].pt])
		for i in range(len(keypoints)):
			x = np.append(x, [keypoints[i].pt], axis=0)
		x = x[1:len(x)]
		bandwidth = estimate_bandwidth(x, quantile=0.5, n_samples=len(x))
		ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
		ms.fit(x)
		labels = ms.labels_
		#cluster_centers = ms.cluster_centers_
		labels_unique = np.unique(labels)
		n_clusters_ = len(labels_unique)
		print("number of estimated clusters : %d" % n_clusters_)
		s = [None] * n_clusters_
		for i in range(n_clusters_):
			l = ms.labels_
			d, = np.where(l == i)
			print(d.__len__())
			s[i] = list(keypoints[xx] for xx in d)

		#des2_ = des2
        
		for idx in range(len(self.blocks_list)):
			kp1, d1 = self.unpackKeypoints(self.keypoints_descriptors[idx])
			for i in range(n_clusters_):
				kp2 = s[i]
				l = ms.labels_
				d, = np.where(l == i)
				des2 = descriptors[d, ]
				flann_params = dict(algorithm = 1, trees = 1)
				matcher = cv.FlannBasedMatcher(flann_params, {})
				#matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
				#matcher = cv.BFMatcher(cv.NORM_L2)
				des2 = np.float32(des2)
				matches = matcher.knnMatch(d1, trainDescriptors = des2, k = 2)
    
				# store all the good matches as per Lowe's ratio test.
				good = []
				for m,n in matches:
					if m.distance < 0.8*n.distance:
						good.append(m)
    
				if len(good)>3:
					src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
					dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
					M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2)
    
					if M is None:
						print ("No Homography")
					else:
						matchesMask = mask.ravel().tolist()
    
					h,w = 50,50
					pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
					try:
						dst = cv.perspectiveTransform(pts,M)
						self.target = cv.polylines(self.target,[np.int32(dst)],True,(0, 255, 0),3, cv.LINE_AA)
						print(idx)
						
					except:
						print ("Não deu para fazer a perspectiva") 
    
				else:
					print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
					
				matchesMask = None

		#self.target_features = self.packKeypoints(keypoints, descriptors)
		plt.imshow(self.target, 'gray'), plt.show()
		
	def process(self, target):
		
		self.loadBlocks()
		self.loadTarget(target)
		#self.matchBlocks()

	def packKeypoints(self,keypoints, descriptors):

		i = 0
		temp_array = []
		for point in keypoints:	   			
			temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, descriptors[i])     
			i += 1	        
			temp_array.append(temp)
		return temp_array

	def unpackKeypoints(self,array):

	    keypoints = []
	    descriptors = []
	    for point in array:
	    	temp_feature = cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
	    	temp_descriptor = point[6]
	    	keypoints.append(temp_feature)
	    	descriptors.append(temp_descriptor)
	    return keypoints, np.array(descriptors)
