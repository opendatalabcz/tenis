from joblib import dump, load
import math
import copy
import pandas as pd
import os
import numpy as np
import cv2
from joblib import dump, load
import sys
sys.path.append('.')
from yolo3.utils.bbox import BoundBox


in_landings_categories = ['LFI', 'LNI','SLFI','SLNI']

out_landings_categories = ['LFO', 'LNO','SLNO','SLFO', 'LNF', 'LNF']

hit_categories = ['HFL','HFR',
                  'HNR','HNL',
                  ]

serve_categories = ['SFL','SFR','SNL','SNR']




def fix_trajectory_to_court(court_points_per_image, trajectory):
  """
    Description:
      Fix trajectory to court point, so during movement of camera trajectory stays fixed to the court
    Parameters:
      court_points_per_image        (list(tuple(int,int)))        : court points found in image
      trajectory                    (list(int,int,int))           : trajectory represented as frame,x,y       
    Returns:
      fixated_trajectory            (list(int,int,int))           : fixated trajectory represented as frame,x,y 
  """

  fixated_trajectory = []
  
  for i in range(len(trajectory)):
    frame = trajectory[i][0]
    bottom_left_corner = (court_points_per_image[frame][3][0],court_points_per_image[frame][3][1])
    new_x = trajectory[i][1] - bottom_left_corner[0]
    new_y = trajectory[i][2] - bottom_left_corner[1]
    fixated_trajectory.append((frame,new_x,new_y))
  return fixated_trajectory
  
  
  
def visualize(images, events_detections_in_trajectory, speed_of_serve, trajectory, court_points_per_image):
  """
    Description:
      Draw trajectory with positon of events on images. On top of image writes last three event names.
    Parameters:
      images                              (list((numpy.array(width,height, depth))))        : list of BGR images
      events_detections_in_trajectory     (list(int,string))                                : list of annotated events, format frame,event
      speed_of_serve                      (int)                                             : speed of serve in km/h
      trajectory                          (list(int,int,int))                               : trajectory represented as frame,x,y 
      court_points_per_image              (list(tuple(int,int)))                            : court points found in image  
    Returns:
      images                              (list((numpy.array(width,height, depth))))        : list of BGR images with visualized trajectory and events
  """
  NUMBER_OF_DETECTION_ON_ONE_FRAME = 10
  events_txt = ''
  events_list = []
  events_list.append('')
  events_list.append('')

  events_detections_in_trajectory_indexes = []
  events_detections_in_trajectory_classification = []
  for index,event_type in events_detections_in_trajectory:
    events_detections_in_trajectory_indexes.append(index)
    events_detections_in_trajectory_classification.append(event_type)

  for i in range(len(trajectory)):
    frame = trajectory[i][0]
    x = int(trajectory[i][1])
    y = int(trajectory[i][2])
    trajectory[i] = (frame,x,y)
    
  trajectory = fix_trajectory_to_court(court_points_per_image, trajectory)  
  
  
  for i in range(0,len(trajectory)-1):#over detections, trajectory index
    if trajectory[i][0] in events_detections_in_trajectory_indexes:
      event_id = events_detections_in_trajectory_indexes.index(trajectory[i][0])
      detection_class = events_detections_in_trajectory_classification[event_id] 
      if detection_class in serve_categories:
        detection_class_to_write = detection_class + ' ' + str(speed_of_serve) + 'km/h'
      else:
        detection_class_to_write = detection_class      
      events_list.append(detection_class_to_write)
      events_txt = events_list[-3] + ' | ' + events_list[-2] +  ' | ' + events_list[-1]
    for k in range(trajectory[i][0],trajectory[i+1][0]): #over frames between two detections,
      bottom_left_corner = (court_points_per_image[k][3][0],court_points_per_image[k][3][1])
      for j in range(max(0,i-NUMBER_OF_DETECTION_ON_ONE_FRAME),i):#number of detections backwards in on frame
        centroid = (trajectory[j][1] + bottom_left_corner[0], trajectory[j][2] + bottom_left_corner[1])
        centroid_next = (trajectory[j+1][1] + bottom_left_corner[0] ,trajectory[j+1][2] + bottom_left_corner[1])
        images[k] = cv2.line(images[k],(centroid[0],centroid[1]), (centroid_next[0],centroid_next[1]), (255,255,255),2) #line between detections
        images[k] = cv2.putText(images[k], events_txt, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if trajectory[j][0] in events_detections_in_trajectory_indexes:
          event_id = events_detections_in_trajectory_indexes.index(trajectory[j][0])
          detection_class = events_detections_in_trajectory_classification[event_id]
          if detection_class in in_landings_categories:
            images[k] = cv2.circle(images[k], centroid, 2, (0,255,0), 5)  
          elif detection_class in out_landings_categories:
            images[k] = cv2.circle(images[k], centroid, 2, (0,0,255), 5)  
          elif detection_class in hit_categories or detection_class in serve_categories:
            images[k] = cv2.circle(images[k], centroid, 2, (0,255,255), 5)
            
            
  return images