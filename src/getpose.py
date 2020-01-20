#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import os

bridge = CvBridge()


def load_image(name):
  script_path = os.path.dirname(os.path.realpath(__file__))

  return cv.imread(script_path + '/data/' + name, cv.IMREAD_GRAYSCALE)

img_object = load_image('obj.jpg')

def image_callback(ros_image):
  global bridge
  #-- convert ros_image into an opencv-compatible image
  try:
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
  except CvBridgeError as e:
    print(e)
  frame = cv_image

  minHessian = 400

  detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
  keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)

  img_scene = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

  if img_object is None or img_scene is None:
    print('Could not open or find the images!')
    exit(0)
  #-- Detect the keypoints using SURF Detector, compute the descriptors
  minHessian = 400
  keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
  ratio_thresh = 0.6
  good_matches = []
  img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
  if keypoints_scene is None or descriptors_scene is None:
    print('Nonetype scene attributes!')
    cv.drawMatches(img_object, keypoints_obj, frame, keypoints_scene, good_matches, img_matches, flags=2  )
  else:
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    #-- Get good matches
    if len(descriptors_scene) < 3:
      print('Number of scene descriptors is too low!')
    else:
      knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
      for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
          good_matches.append(m)

    #-- Create data for Homography search
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
      obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
      obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
      scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
      scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    #-- Get Homography data
    if len(good_matches) > 1:
      H, _ =  cv.findHomography(obj, scene, cv.RANSAC)
      if H is None:
        print('No Homography')
      else:
        obj_corners = np.empty((4,1,2), dtype=np.float32)
        obj_corners[0,0,0] = 0
        obj_corners[0,0,1] = 0
        obj_corners[1,0,0] = img_object.shape[1]
        obj_corners[1,0,1] = 0
        obj_corners[2,0,0] = img_object.shape[1]
        obj_corners[2,0,1] = img_object.shape[0]
        obj_corners[3,0,0] = 0
        obj_corners[3,0,1] = img_object.shape[0]
        scene_corners = cv.perspectiveTransform(obj_corners, H)
        
        #-- Show object edges
        h, w = img_object.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,H)
        frame = cv.polylines(frame,[np.int32(dst)],True,(255,255,255),3, cv.LINE_AA)
        #-- Put object data
        ow = 148
        oh = 223
        model = np.array([
          (0.0, 0.0, 0.0),
          (0.0, oh, 0.0 ),
          (ow, oh, 0.0),
          (ow, 0.0, 0.0)])
        #-- Put camera data
        focal = 820
        cx = 290
        cy = 234
        dist = np.zeros((4,1))
        camera_matrix = np.array(
                        [[focal, 0, cx],
                        [0, focal, cy],
                        [0, 0, 1]], dtype = "double")
        (success, rotation_vector, translation_vector) = cv.solvePnP(model, dst, camera_matrix, dist, flags=cv.SOLVEPNP_ITERATIVE)
        if success:
          print("tv = ", translation_vector)
          print("rv = ", rotation_vector )
    else:
      print('Good matches length < 1')

    #-- Create two images
    
    cv.drawMatches(img_object, [], frame, [], [], img_matches, flags=0)

    #-- Show image
    font = cv.FONT_HERSHEY_SIMPLEX
    text = 'Matches '+str(len(good_matches))
    place = (10+img_object.shape[1],frame.shape[0]-30)
    cv.putText(img_matches,text, place, font, 1,(255,255,255),2,cv.LINE_AA)
    cv.imshow('Object detection', img_matches)
    cv.waitKey(10)

def main(args):
  rospy.init_node('image_converter', anonymous=True)

  image_sub = rospy.Subscriber("image",Image, image_callback)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
