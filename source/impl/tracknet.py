import cv2
import numpy as np
import os
import copy
import sys
sys.path.append('.')
from TrackNet.Code_Python3.TrackNet_Three_Frames_Input.Models.TrackNet import TrackNet 
from joblib import dump, load


#parameters from authors of Tracknet implementation https://nol.cs.nctu.edu.tw:234/open-source/TrackNet
n_classes = 256
input_width =  640
input_height = 360

MIN_DIST_BETWEEN_DETECTIONS = 7 

def get_tracknet_model(path_to_weights):
  """
    Description:
      Compile TrackNet model
    Parameters:
      path_to_weights              (string)         : path to tracknet weights
    Returns:
      m                           (keras model)    : compiled keras model
  """
  m = TrackNet( n_classes , input_height=input_height, input_width=input_width ) 
  m.compile(loss='categorical_crossentropy', optimizer= 'adadelta', metrics=['accuracy']) 
  m.load_weights( path_to_weights )
  return m
    
def prepare_images(images):
  """
    Description:
      Convert images to assumed format
    Parameters:
      images                          (list((numpy.array(width,height, depth))))        : list of BGR images
    Returns:
      images                          (list((numpy.array(width,height, depth))))        : list of BGR images in assumed format
  """
  images_prepared = []
  for image in images:
    img = cv2.resize(image, (640,360))
    img = img.astype(np.float32)
    images_prepared.append(img)
  return images_prepared
    
def get_detections_in_images(images, model):
  """
    Description:
      Find tennis ball detections in images
    Parameters:
      images                          (list((numpy.array(width,height, depth))))        : list of BGR images
      m                               (keras model)                                     : compiled keras model of TrackNet
    Returns:
      detections_in_images            (list(list(tuple[int,int])))                      : list of lists of detections in frames, [#frame][#detection], format of detection is x,y
  """
  output_height, output_width = images[0].shape[0:2]
  
  len_of_clip = len(images)
  detections_in_images = [0] * len_of_clip
  for i in range(len_of_clip):
    detections_in_images[i] = []
    
  img3 = cv2.resize(images[0], (640,360)).astype(np.float32)
  img2 = cv2.resize(images[1], (640,360)).astype(np.float32)
  for i in range(2,len(images)):
    img1 = cv2.resize(images[i], (640,360)).astype(np.float32)
    
    X =  np.concatenate((img1, img2, img3),axis=2)
    X = np.rollaxis(X, 2, 0)
    pr = model.predict( np.array([X]) )[0]
    pr = pr.reshape(( input_height ,  input_width , n_classes ) ).argmax( axis=2 )
    pr = pr.astype(np.uint8)
    print(f"tracknet img {i}")
    #reshape the image size as original input image
    heatmap = cv2.resize(pr  , (output_width, output_height ))
    #heatmap is converted into a binary image by threshold method.
    ret,heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=MIN_DIST_BETWEEN_DETECTIONS,param1=50,param2=2,minRadius=2,maxRadius=7)# pro minDist < 3 detekuje 4 body pro micek, spatny, TDLA expn diameter max 7 asi
    #in circles is it x,y,radius
    if (circles is not None):
      for circle in circles[0]:
          detections_in_images[i].append((circle[0],circle[1]))
    
    img3 = img2
    img2 = img1
    
  return detections_in_images
    
    
    
    