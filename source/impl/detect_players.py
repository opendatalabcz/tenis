
import cv2
import numpy as np
import os
from os.path import isfile,join
import sys

import os
import argparse
import json
import cv2

from keras.models import load_model
from tqdm import tqdm
import numpy as np
import copy
from joblib import dump, load
import math
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from scipy.ndimage.filters import uniform_filter1d

sys.path.append('.')
from utils import compare_file_paths,get_end_point_from_vector_and_point, get_intersection_of_two_lines_segments, get_distance_point_line, add, scale, distance, unit, vector, length, dot, is_inside_area
from yolo3.utils.utils import get_yolo_boxes, makedirs
from yolo3.utils.bbox import draw_boxes, BoundBox
from functools import cmp_to_key



# yolov3 source https://github.com/experiencor/keras-yolo3
def get_yolov3_model(config_path):
  """
    Description:
      Compile YOLOv3 model
    Parameters:
      config_path                                (string)         : path to yolov3 configuration file
    Returns:
      infer_model                                (keras model)    : compiled keras model
  """
  with open(config_path) as config_buffer:    
    config = json.load(config_buffer)
  os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
  infer_model = load_model(config['train']['saved_weights_name'])
  return infer_model
  
  
def yolo_predict(infer_model, config_path, images, threshold_confidence):
  """
    Description:
      Predict bounding boxes for humans in images
    Parameters:
      infer_model                     (keras model)                                     : compiled keras model
      config_path                     (string)                                          : path to yolov3 configuration file
      images                          (list((numpy.array(width,height, depth))))        : list of BGR images
      threshold_confidence            (float)                                           : threshold for confidence for detection, 0-1
    Returns:
      boxes_all                       (list(list(yolo3.utils.bbox.BoundBox)))           : list of bounding boxes for each image
  """
  with open(config_path) as config_buffer:    
    config = json.load(config_buffer)


  ###############################
  #   Set some parameter
  ###############################       
  net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
  obj_thresh, nms_thresh = threshold_confidence, 0.3

  boxes_all = []
  for image in images:
    image_height, image_width = image.shape[0:2]
    boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
    print(f"yolov3 processing image {len(boxes_all)}")
    for i in range(len(boxes)-1,-1,-1):
      if boxes[i].classes[0] < threshold_confidence or boxes[i].xmax > image_width or boxes[i].xmin < 0 or boxes[i].ymin < 0 or boxes[i].ymax > image_height:
        boxes.pop(i)
    boxes_all.append(boxes)

  return boxes_all

def score_human_position(player_position, bottom_right_corner, bottom_left_corner, top_right_corner, top_left_corner, image_width, image_height):
  """
    Description:
      Score position of player based on court information
    Parameters:
      player_position                 ((int,int))                         : player position, x,y
      bottom_right_corner             (int)                               : bottom_right_corner of area of interest
      bottom_left_corner              (int)                               : bottom_left_corner of area of interest
      top_right_corner                (int)                               : top_right_corner of area of interest
      top_left_corner                 (int)                               : top_left_corner of area of interest
      image_width                     (int)                               : image width in pixels
      image_height                    (int)                               : image heigth in pixels
    Returns:
                                      (float)                             : 0 if player in area of interest, otherwise smaller distance from player to bottom_left_corner - bottom_right_corner and  top_left_corner-top_right_corner lines
  """
  if is_inside_area(player_position, bottom_right_corner, bottom_left_corner, top_right_corner, top_left_corner, image_width, image_height) == 'in':
    return 0
  else:
    return min(get_distance_point_line(player_position, bottom_left_corner, bottom_right_corner),get_distance_point_line(player_position, top_left_corner, top_right_corner))#vzdalenosti od base lines

def get_players_in_image(boxes,bottom_right_corner,bottom_left_corner,top_right_corner,top_left_corner,image_height,image_width):
  """
    Description:
      Choose two human detections as players for each image
    Parameters:
      boxes                           (list(list(yolo3.utils.bbox.BoundBox)))             : list of bounding boxes for each image
      bottom_right_corner             (int)                                               : bottom_right_corner of area of interest
      bottom_left_corner              (int)                                               : bottom_left_corner of area of interest
      top_right_corner                (int)                                               : top_right_corner of area of interest
      top_left_corner                 (int)                                               : top_left_corner of area of interest
      image_width                     (int)                                               : image width in pixels
      image_height                    (int)                                               : image heigth in pixels
    Returns:
                                      (float)                                             : 0 if player in area of interest, otherwise smaller distance from player to bottom_left_corner - bottom_right_corner and  top_left_corner-top_right_corner lines
  """

  if len(boxes) == 0:
    return (None,None)
  player1_index = 0
  player1_score = float('inf')
  player2_index = 0
  player2_score = float('inf')

  for j in range(len(boxes)):
    box = boxes[j]
    x_min = box.xmin
    x_max = box.xmax
    y_min = box.ymin
    y_max = box.ymax
    player_position1 = ((x_min + x_max)/2,y_max)
    player_position2 = ((x_min + x_max)/2,y_min)
    score1 = score_human_position(player_position1,bottom_right_corner,bottom_left_corner,top_right_corner,top_left_corner,image_width,image_height)
    score2 = score_human_position(player_position2,bottom_right_corner,bottom_left_corner,top_right_corner,top_left_corner,image_width,image_height)
    score = min(score1,score2)
    if score < player1_score:
      player2_index = player1_index
      player1_index = j
      player2_score = player1_score
      player1_score = score
    elif score < player2_score:
      player2_index = j
      player2_score = score

  if (player1_score != float('inf')) and (player2_score != float('inf')):
    if boxes[player1_index].ymin > boxes[player2_index].ymin:
      return (boxes[player2_index],boxes[player1_index])
    else:
      return (boxes[player1_index],boxes[player2_index])
  else:
    return ([],[])    
    

def find_outliers(trajectory_to_filtr, min_distance_to_start_process):
  """
    Description:
      Find outliers in time series and predict their true location by linear regression
    Parameters:
      trajectory_to_filtr               (list(int))             : time series
      min_distance_to_start_process     (float)                 : minimal distance between neighboring values to start finding outliers
    Returns:
      outliers_id                       (list(int))             : indexes of outliers
      trajectory_to_filtr               (list(int))             : time series with outliers replaced by linear regression predicitons
  """
  outliers_id = []
  dists = []
  dists.append(0)
  for i in range(1,len(trajectory_to_filtr)):
    dists.append(int(trajectory_to_filtr[i] - trajectory_to_filtr[i-1]))
  dists = np.array(dists)

  if max(dists) < min_distance_to_start_process:
    return [], trajectory_to_filtr


  anomalies = []
  # Set upper and lower limit to 3 standard deviation
  random_data_std = np.std(dists)
  random_data_mean = np.mean(dists)
  anomaly_cut_off3 = random_data_std * 3
  anomaly_cut_off2 = random_data_std * 2

  lower_limit  = random_data_mean - anomaly_cut_off3
  upper_limit = random_data_mean + anomaly_cut_off3

  repaired = []
  while True:
    anomalies = []
    for i in range(len(dists)):
      outlier = dists[i]
      if outlier > upper_limit or outlier < lower_limit:
          if i not in repaired:
            anomalies.append(i)
            break 
    repaired = repaired + anomalies
    for id in anomalies:
      low = max(0,id-10)
      high = min(len(trajectory_to_filtr),id+10)
      outliers_id.append(id)

      y = np.array([i for i in range(low,high)])
      y = np.delete(y,10)
      model = LinearRegression()
      to_fit = np.append(np.array(trajectory_to_filtr[low:id]),np.array(trajectory_to_filtr[id+1:high])).reshape(-1, 1)
      model.fit(y.reshape(-1, 1),to_fit )
      
      trajectory_to_filtr[id] = int(model.predict(np.array(id).reshape(1, -1))[0][0])
      
      dists[id] = trajectory_to_filtr[id] - trajectory_to_filtr[id-1]
      if id != (len(trajectory_to_filtr) -1):
        dists[id+1] = trajectory_to_filtr[id+1] - trajectory_to_filtr[id]
    if anomalies == []:
      break
  return outliers_id, trajectory_to_filtr
      
def interpolate_outliers(player_bottom_bb, player_bottom_x, player_bottom_y, player_top_bb, player_top_x, player_top_y,  players_not_detected_indexes):
  """
    Description:
      Find outliers in time series and predict their true location by linear regression
    Parameters:
      player_bottom_bb              (list(yolo3.utils.bbox.BoundBox))             : bounding boxes of bottom player per image
      player_bottom_x               (list(int))                                   : time series of bottom player x coord
      player_bottom_y               (list(int))                                   : time series of bottom player y coord
      player_top_bb                 (list(yolo3.utils.bbox.BoundBox))             : bounding boxes of top player per image
      player_top_x                  (list(int))                                   : time series of top player x coord
      player_top_y                  (list(int))                                   : time series of top player y coord
      players_not_detected_indexes  (list(int))                                   : list of indexes where player not detected
    Returns:
      player_bottom_bb              (list(yolo3.utils.bbox.BoundBox))             : bounding boxes of bottom player per image with repaired outliers
      player_bottom_bb              (list(yolo3.utils.bbox.BoundBox))             : bounding boxes of bottom player per image with repaired outliers
  """
  
  min_distance_to_start_process = 10
  for i in range(len(players_not_detected_indexes)-1,-1,-1):
    player_bottom_x.pop(players_not_detected_indexes[i]) 
    player_bottom_y.pop(players_not_detected_indexes[i])
    player_top_x.pop(players_not_detected_indexes[i])
    player_top_y.pop(players_not_detected_indexes[i])
    player_top_bb.pop(players_not_detected_indexes[i])
    player_bottom_bb.pop(players_not_detected_indexes[i])
    
  player_bottom_x_outliers, player_bottom_x = find_outliers(player_bottom_x,min_distance_to_start_process)
  player_bottom_y_outliers, player_bottom_y = find_outliers(player_bottom_y,min_distance_to_start_process)
  player_top_x_outliers, player_top_x = find_outliers(player_top_x,min_distance_to_start_process)
  player_top_y_outliers, player_top_y = find_outliers(player_top_y,min_distance_to_start_process)
  outliers_player_top =  player_top_x_outliers + player_top_y_outliers
  outliers_player_bottom = player_bottom_x_outliers + player_bottom_y_outliers 
  
  outliers_player_top = sorted(outliers_player_top)
  outliers_player_bottom = sorted(outliers_player_bottom)
  
  #shift BB to position from linear regression prediction
  for outlier in outliers_player_bottom:
    x_move = player_bottom_x[outlier] - player_bottom_x[outlier-1]
    y_move = player_bottom_y[outlier] - player_bottom_y[outlier-1]
    box = copy.deepcopy(player_bottom_bb[outlier-1])
    if outlier in player_bottom_x_outliers and outlier in player_bottom_y_outliers: 
      box.xmax = box.xmax + x_move
      box.xmin = box.xmin + x_move
      box.ymax = box.ymax + y_move
      box.ymin = box.ymin + y_move
    elif outlier in player_bottom_x_outliers:
      box.xmax = box.xmax + x_move
      box.xmin = box.xmin + x_move
    elif outlier in player_bottom_y_outliers:
      box.ymax = box.ymax + y_move
      box.ymin = box.ymin + y_move

    player_bottom_bb[outlier] = box

  for outlier in outliers_player_top:
    x_move = player_top_x[outlier] - player_top_x[outlier-1]
    y_move = player_top_y[outlier] - player_top_y[outlier-1]
    box = copy.deepcopy(player_top_bb[outlier-1])
    if outlier in player_top_x_outliers and outlier in player_top_y_outliers: 
      box.xmax = box.xmax + x_move
      box.xmin = box.xmin + x_move
      box.ymax = box.ymax + y_move
      box.ymin = box.ymin + y_move
    elif outlier in player_top_x_outliers:
      box.xmax = box.xmax + x_move
      box.xmin = box.xmin + x_move
    elif outlier in player_top_y_outliers:
      box.ymax = box.ymax + y_move
      box.ymin = box.ymin + y_move

    player_top_bb[outlier] = box
    
  for i in range(len(players_not_detected_indexes)-1,-1,-1): #not detected, insert dummy box
    bb = BoundBox(0,0,0,0)
    player_top_bb.insert(players_not_detected_indexes[i], bb)
    player_bottom_bb.insert(players_not_detected_indexes[i], bb)
    
  return player_top_bb, player_bottom_bb
  
def get_players_bounding_boxes_in_images(model, images, court_points_per_images, config_path):
  """
    Description:
      Predict bounding boxes for players in images
    Parameters:
      model                           (keras model)                                     : compiled keras model
      images                          (list((numpy.array(width,height, depth))))        : list of BGR images
      court_points_per_images         (list(tuple(int,int)))                            : court points found in images           
      config_path                     (string)                                          : path to yolov3 configuration file
    Returns:
      player_candidates               (dict(list(yolo3.utils.bbox.BoundBox)))           : list of bounding boxes for top and bottom player per images, 'player_top' and 'player_bottom' are keys to dict
  """

  threshold_confidence = 0.3 #smaller confidence because yolo3 not trained on tennis domain, so players in strange positions detected with low confidence
  boxes_all = yolo_predict(model, config_path,images,threshold_confidence)
  
  player_top_bb = []
  player_bottom_bb = []
  player_bottom_x = []
  player_bottom_y = []
  player_top_x = []
  player_top_y = []

  are_players_detected = []
  players_not_detected_indexes = []

  image_height,image_width = images[0].shape[0:2]
  for i in range(len(images)):
    boxes = boxes_all[i]
    if court_points_per_images[i] is None or len(boxes) == 0: #court not detected so do not detect players
      player_bottom_x.append(0)
      player_bottom_y.append(0)
      player_top_x.append(0)
      player_top_y.append(0)
      bb = BoundBox(0,0,0,0)
      player_top_bb.append(bb)
      player_bottom_bb.append(bb)
      are_players_detected.append(False)
      players_not_detected_indexes.append(i)
      continue
    
    bottom_right_corner = court_points_per_images[i][0]
    bottom_left_corner = court_points_per_images[i][3]
    top_right_corner = court_points_per_images[i][10]
    top_left_corner = court_points_per_images[i][13]

    #outliers filtration by center of BB, because center of BB is not changing so much when size of BB changes during player movement
    player_top_box, player_bottom_box = get_players_in_image(boxes,bottom_right_corner,bottom_left_corner,top_right_corner,top_left_corner,image_height,image_width)
    player_bottom_x.append(int((player_bottom_box.xmax + player_bottom_box.xmin) / 2))
    player_bottom_y.append(int((player_bottom_box.ymax + player_bottom_box.ymin) / 2))
    player_top_x.append(int((player_top_box.xmax + player_top_box.xmin) / 2))
    player_top_y.append(int((player_top_box.ymax + player_top_box.ymin) / 2))
    player_top_bb.append(player_top_box)
    player_bottom_bb.append(player_bottom_box)
    are_players_detected.append(True)
  
  player_top_bb, player_bottom_bb = interpolate_outliers(player_bottom_bb, player_bottom_x, player_bottom_y,player_top_bb, player_top_x, player_top_y,  players_not_detected_indexes)
  
  player_candidates = {
    'player_bottom' : [],
    'player_top'    : [],
  }

  for i in range(len(player_top_bb)):
    player_candidates['player_top'].append(player_top_bb[i])
    player_candidates['player_bottom'].append(player_bottom_bb[i])
    
  return player_candidates
  