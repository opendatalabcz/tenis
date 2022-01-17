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
from utils import Path, unit, get_angle2, get_angle_vectors, get_intersection_of_two_lines_segments,get_distance_point_line, compute_distance, is_inside_area, get_end_point_from_vector_and_point, get_center_from_bounding_box, get_bottom_center_from_bounding_box, rotate_vector, get_end_point_from_vector_and_point, get_corner_points_for_tennis_court
from yolo3.utils.bbox import BoundBox
from functools import cmp_to_key

     
def filter_events_by_predefined_rules(i,trajectory,events_trajectory_indexes, events_indexes_classificated, players_detections, corner_points_per_images, image_width, image_height, keep_net_area_events = False):
  """
    Description:
      Filter detections categories by predefined rules
    Parameters:
      i                             int                                 : index of event to check
      trajectory                    list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      events_trajectory_indexes     list(int)                           : list of events indexes to trajectory
      events_indexes_classificated  list(string)                        : list of events categories
      players_detections            list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images      list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      toleration_shift              int                                 : try to shift the ball in y direction. 
      image_width                   int                                 :image widht
      image_height                  int                                 :image height
    Returns:
      events_trajectory_indexes     list(int)                           : if event is removed, index is removed from list
      events_indexes_classificated list(string)                         : if event is removed, then is removed from list, if it is changed, then it is changed in the list
  """
  
  
  events_trajectory_indexes_reclassified_as_land = None 
  events_trajectory_indexes_reclassified_as_hit_near = None
  events_trajectory_indexes_reclassified_as_hit_far = None
  events_indexes_to_remove = None
  current_index = events_trajectory_indexes[i]
  current_frame_index = trajectory[current_index][0]

  if events_indexes_classificated[i] == 'land_near':
    possible_landing_spot = can_be_landing_spot(i,trajectory,events_trajectory_indexes,image_width, image_height)
    if possible_landing_spot == False:
      if can_be_hit_spot_near(i,trajectory,events_trajectory_indexes, players_detections,corner_points_per_images,image_width, image_height):
        events_trajectory_indexes_reclassified_as_hit_near = events_trajectory_indexes[i]
      elif can_be_hit_spot_far(i,trajectory,events_trajectory_indexes, players_detections,corner_points_per_images,image_width, image_height):
        events_trajectory_indexes_reclassified_as_hit_far = events_trajectory_indexes[i]
      else:
        events_indexes_to_remove = i
        
        
  if events_indexes_classificated[i] == 'land_far':
    possible_landing_spot = can_be_landing_spot(i,trajectory,events_trajectory_indexes,image_width, image_height)
    if possible_landing_spot == False:
      if can_be_hit_spot_far(i,trajectory,events_trajectory_indexes, players_detections,corner_points_per_images,image_width, image_height):
        events_trajectory_indexes_reclassified_as_hit_far = events_trajectory_indexes[i]
      elif can_be_hit_spot_near(i,trajectory,events_trajectory_indexes, players_detections,corner_points_per_images,image_width, image_height):
        events_trajectory_indexes_reclassified_as_hit_near = events_trajectory_indexes[i]
      else:
        events_indexes_to_remove = i
        
        
  elif events_indexes_classificated[i] == 'hit_far' or events_indexes_classificated[i] == 'hit_near':
    index_of_event = i
    current_trajectory_index = events_trajectory_indexes[index_of_event]
    current_frame_index = trajectory[current_trajectory_index][0]
    if events_indexes_classificated[i] == 'hit_near':
      if not can_be_hit_spot_near(i,trajectory,events_trajectory_indexes, players_detections,corner_points_per_images,image_width, image_height): #nemusim testovat hit spot far, proto bliz near hraco
        possible_landing_spot = can_be_landing_spot(index_of_event,trajectory,events_trajectory_indexes,image_width, image_height)
        if not possible_landing_spot:
          events_indexes_to_remove = index_of_event
        else:
          events_trajectory_indexes_reclassified_as_land = current_trajectory_index   
    #closer to top player
    else:
      current_frame_index = trajectory[current_index][0]
      if not can_be_hit_spot_far(i,trajectory,events_trajectory_indexes, players_detections,corner_points_per_images,image_width, image_height):
        possible_landing_spot = can_be_landing_spot(index_of_event,trajectory,events_trajectory_indexes,image_width, image_height)
        if not possible_landing_spot:
          events_indexes_to_remove = index_of_event
        else:
          events_trajectory_indexes_reclassified_as_land = current_trajectory_index

  return events_indexes_to_remove,events_trajectory_indexes_reclassified_as_land, events_trajectory_indexes_reclassified_as_hit_near, events_trajectory_indexes_reclassified_as_hit_far

  
  
def remove_events(events_trajectory_indexes, events_indexes_classificated, events_to_remove, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height):
  """
    Description:
      Remove events and reclassify neighbor events
    Parameters:
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      events_indexes_classificated                        list(string)                        : list of events categories
      events_to_remove                                    list(int)                           : list of events indexes to remove
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      homographies_image_to_artificial_court_per_frame    list(homography)                    : list of homographies per image from cv2.findHomography()
      points_artificial_court                             list(list(tuple[int,int]))          : list of court points of artificial court with real world dimensions
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
      events_trajectory_indexes     list(int)                           : if event is removed, index is removed from list
      events_indexes_classificated list(string)                         : if event is removed, then is removed from list, if it is changed, then it is changed in the list
  """
  
  for i in range(len(events_to_remove)-1,-1,-1):
    event_index = events_to_remove[i]
    
    if event_index == 0: 
      events_trajectory_indexes.pop(event_index)
      events_indexes_classificated.pop(event_index)
      
      if event_index <= (len(events_trajectory_indexes)-1):
        event_index_to_correct1 = event_index
        event_type = classify_event(event_index_to_correct1,events_trajectory_indexes, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court)
        events_indexes_classificated[event_index_to_correct1] = event_type
        
    elif event_index == (len(events_trajectory_indexes)-1): 
      events_trajectory_indexes.pop(event_index)
      events_indexes_classificated.pop(event_index)
      
      event_index_to_correct1 = event_index-1 
      event_type = classify_event(event_index_to_correct1,events_trajectory_indexes, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court)
      events_indexes_classificated[event_index_to_correct1]= event_type
    
    else:
      events_trajectory_indexes.pop(event_index)
      events_indexes_classificated.pop(event_index)
      
      event_index_to_correct1 = event_index-1 
      event_index_to_correct2 = event_index 
      
      event_type = classify_event(event_index_to_correct1,events_trajectory_indexes, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court)
      events_indexes_classificated[event_index_to_correct1] = event_type
      
      event_type = classify_event(event_index_to_correct2,events_trajectory_indexes, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court)
      events_indexes_classificated[event_index_to_correct2] = event_type

  return events_trajectory_indexes,events_indexes_classificated


def can_be_hit_spot_near(index,trajectory,events_trajectory_indexes, players_detections,corner_points_per_images,image_width, image_height):
  """
    Description:
      Check if event can be hit spot of bottom player
    Parameters:
      index                                               int                                 : index of event to check
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
                                                          bool                                : True if it can be hit spot of bottom player, false otherwise
  """
  current_trajectory_index = events_trajectory_indexes[index]
  current_frame_index = trajectory[current_trajectory_index][0]
  player_top_bb = players_detections['player_top'][current_frame_index]
  player_top_center_of_bb = (int((player_top_bb.xmax + player_top_bb.xmin)/2),int((player_top_bb.ymax + player_top_bb.ymin)/2))
  player_bottom_bb = players_detections['player_bottom'][current_frame_index]
  player_bottom_center_of_bb = (int((player_bottom_bb.xmax + player_bottom_bb.xmin)/2),int((player_bottom_bb.ymax + player_bottom_bb.ymin)/2))
  
  if index != 0:
    most_backward_index = events_trajectory_indexes[index-1] 
  else:
    most_backward_index = current_trajectory_index - 3
    
  if index != (len(events_trajectory_indexes)-1):
    most_forward_index = events_trajectory_indexes[index+1] 
  else:
    most_forward_index = len(trajectory)-1
  
  ball_position = (trajectory[current_trajectory_index][1],trajectory[current_trajectory_index][2])
  dist_to_top_player = compute_distance(player_top_center_of_bb, ball_position)
  dist_to_bottom_player = compute_distance(player_bottom_center_of_bb, ball_position)
  
  if dist_to_top_player > dist_to_bottom_player:
    velocity_before_event_y = trajectory[current_trajectory_index][2] - trajectory[most_backward_index][2]
    velocity_after_event_y = trajectory[most_forward_index][2] - trajectory[current_trajectory_index][2]

    if (velocity_before_event_y < 0 and velocity_after_event_y < 0) or (velocity_before_event_y > 0 and velocity_after_event_y < 0):
      return True
      
    elif (velocity_before_event_y <= 0 and velocity_after_event_y > 0): 
      for i in range(current_trajectory_index, most_forward_index):
         ball_position = (trajectory[i][1],trajectory[i][2])
         current_frame_index = trajectory[i][0]
         res = classify_landing_spot(ball_position,corner_points_per_images,current_frame_index,image_width, image_height)
         if res == 'land_far':
          return True
  return False
  
  
  
def can_be_hit_spot_far(index,trajectory,events_trajectory_indexes, players_detections,corner_points_per_images,image_width, image_height):
  """
    Description:
      Check if event can be hit spot of top player
    Parameters:
      index                                               int                                 : index of event to check
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
                                                          bool                                : True if it can be hit spot of top player, false otherwise
  """
    
  current_trajectory_index = events_trajectory_indexes[index]
  current_frame_index = trajectory[current_trajectory_index][0]
  player_top_bb = players_detections['player_top'][current_frame_index]
  player_top_center_of_bb = (int((player_top_bb.xmax + player_top_bb.xmin)/2),int((player_top_bb.ymax + player_top_bb.ymin)/2))
  player_bottom_bb = players_detections['player_bottom'][current_frame_index]
  player_bottom_center_of_bb = (int((player_bottom_bb.xmax + player_bottom_bb.xmin)/2),int((player_bottom_bb.ymax + player_bottom_bb.ymin)/2))
  
  if index != 0:
    most_backward_index = events_trajectory_indexes[index-1] 
  else:
    most_backward_index = current_trajectory_index - 3
    
  if index != (len(events_trajectory_indexes)-1):
    most_forward_index = events_trajectory_indexes[index+1] 
  else:
    most_forward_index = len(trajectory)-1
  
  ball_position = (trajectory[current_trajectory_index][1],trajectory[current_trajectory_index][2])
  dist_to_top_player = compute_distance(player_top_center_of_bb, ball_position)
  dist_to_bottom_player = compute_distance(player_bottom_center_of_bb, ball_position)
  if dist_to_top_player < dist_to_bottom_player:#bottom player hit
    velocity_before_event_y = trajectory[current_trajectory_index][2] - trajectory[most_backward_index][2]
    velocity_after_event_y = trajectory[most_forward_index][2] - trajectory[current_trajectory_index][2]
    position = classify_landing_spot(ball_position,corner_points_per_images,current_frame_index,image_width, image_height) 
    
    if ((velocity_before_event_y <= 0 and velocity_after_event_y > 0) or (velocity_before_event_y <= 0 and velocity_after_event_y < 0)) and position == 'land_far' : 
      return True
     
  return False

def can_be_landing_spot(index,trajectory,events_trajectory_indexes, image_width, image_height):
  """
    Description:
      Check if event can be landing spot
    Parameters:
      index                                               int                                 : index of event to check
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
                                                          bool                                : True if it can be hit spot of top player, false otherwise
  """

  current_index = events_trajectory_indexes[index]
  is_intersection = False
  for i in range(1,3):
    forward_index = current_index + i 
    backward_index = current_index - i
    line1 = ((trajectory[current_index][1],trajectory[current_index][2]), (trajectory[current_index][1],0))
    line2 = ((trajectory[backward_index][1],trajectory[backward_index][2]),(trajectory[forward_index][1],trajectory[forward_index][2]))
    #x coord too close to each other, lenghten little bit
    if abs(trajectory[backward_index][1] - trajectory[forward_index][1]) < 5:
      vector1 = (trajectory[backward_index][1] - trajectory[forward_index][1],trajectory[backward_index][2] - trajectory[forward_index][2]) #to backward
      vector2 = (trajectory[forward_index][1] - trajectory[backward_index][1],trajectory[forward_index][2] - trajectory[backward_index][2]) #to forward
      vector1 = unit(vector1)
      vector2 = unit(vector2)
      line2 = ((trajectory[backward_index][1] + 5*vector1[0] ,trajectory[backward_index][2]  + 5*vector1[1])  , (trajectory[forward_index][1]+ 5*vector2[0],trajectory[forward_index][2] + 5*vector2[1]))
    rotations = [0,3,357]
    vector1 = (0,-1)
    start_point = (trajectory[current_index][1],trajectory[current_index][2])
    lines_path1 = []
    for rotation in rotations:
      vector1_rotated = rotate_vector(vector1,rotation)
      end_point_rot1 = get_end_point_from_vector_and_point(vector1_rotated,start_point,image_width,image_height)
      lines_path1.append((start_point,end_point_rot1))
    for line1 in lines_path1:
      px,py = get_intersection_of_two_lines_segments(line1[0],line1[1],line2[0],line2[1])
      if px != -1 and py != -1:
        is_intersection = True
        break

  return is_intersection
  
def classify_landing_spot(ball_position,corner_points_per_images,current_frame_index,image_width, image_height):
  """
    Description:
      Classify if detection is on top or bottom part of image divided by half of the court extended to the ends of the image
    Parameters:
      ball_position                                       (int,int)                           : x,y location of ball detection
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      current_frame_index                                 int                                 : frame index of detection
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
                                                          string                              : 'land_far' if on top half, 'land_near' if on bottom half
  """
    
  left_half = corner_points_per_images[current_frame_index][16]
  right_half = corner_points_per_images[current_frame_index][14]
  
  vector_left = (left_half[0] - right_half[0], left_half[1] - right_half[1])
  vector_right = (right_half[0] - left_half[0], right_half[1] - left_half[1])

  half_left_end = get_end_point_from_vector_and_point(vector_left, left_half, image_width, image_height)
  half_right_end = get_end_point_from_vector_and_point(vector_right, right_half, image_width, image_height)
  right_top_image_corner =(image_width,0)
  left_top_image_corner = (0,0)
  right_bottom_image_corner = (image_width,image_height)
  left_bottom_image_corner =(0,image_height)

  land_near = is_inside_area(ball_position, right_bottom_image_corner, left_bottom_image_corner, half_right_end, half_left_end, image_width, image_height)
  land_far = is_inside_area(ball_position,half_right_end,half_left_end, right_top_image_corner, left_top_image_corner, image_width, image_height)
  if land_near == 'in':
    return 'land_near'
  elif land_far == 'in':
    return 'land_far'
    
def classify_landing_spot2(trajectory_index,trajectory,current_frame_index,homographies_image_to_artificial_court_per_frame, points_artificial_court):
  """
    Description:
      Classsify landing spot if it landed on top half or bottom half or outside of the court by hommography
    Parameters:
      trajectory_index                                    int                                 : index of detection in trajectory to check
      current_frame_index                                 int                                 : frame index of detection
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      homographies_image_to_artificial_court_per_frame    list(homography)                    : list of homographies per image from cv2.findHomography()
      points_artificial_court                             list(list(tuple[int,int]))          : list of court points of artificial court with real world dimensions
    Returns:
                                                          string                              : 'land_far' if on top half, 'land_near' if on bottom half, 'out'  if outside of the court
  """
  land_near = is_inside_area_by_homography(trajectory_index,trajectory, homographies_image_to_artificial_court_per_frame, points_artificial_court, 'bottom_half')
  land_far = is_inside_area_by_homography(trajectory_index,trajectory, homographies_image_to_artificial_court_per_frame, points_artificial_court, 'top_half')
  if land_near == True:
    return 'land_near'
  elif land_far == True:
    return 'land_far'
  else:
    print(f"out id {trajectory_index}")
    return 'out'

def is_inside_area_by_homography(trajectory_index,trajectory, homographies_image_to_artificial_court_per_frame, points_artificial_court, option):
  """
    Description:
      Classsify landing spot if ball is inside area by homography
    Parameters:
      trajectory_index                                    int                                 : index of detection in trajectory to check
      current_frame_index                                 int                                 : frame index of detection
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      homographies_image_to_artificial_court_per_frame    list(homography)                    : list of homographies per image from cv2.findHomography()
      points_artificial_court                             list(list(tuple[int,int]))          : list of court points of artificial court with real world dimensions
      option                                              string                              : which area check
    Returns:
                                                          bool                                : True if detection inside area, False otherwise
  """
  land_position = (np.float32(trajectory[trajectory_index][1]),np.float32(trajectory[trajectory_index][2]))
  frame_index = trajectory[trajectory_index][0]
  homography_for_land = homographies_image_to_artificial_court_per_frame[frame_index]
  land_position_real = cv2.perspectiveTransform(np.array([[land_position]]), homography_for_land)[0][0]
  
  if option == 'top_half':
    if land_position_real[1] <= points_artificial_court[15][1]:
      return True
  elif option == 'bottom_half':
    if land_position_real[1] > points_artificial_court[15][1]:
      return True
  elif option == 'top_left_serve':
    return is_inside_rectangle(land_position_real,points_artificial_court[16],points_artificial_court[15],points_artificial_court[9],points_artificial_court[8])
  elif option == 'top_right_serve':
    return is_inside_rectangle(land_position_real,points_artificial_court[15],points_artificial_court[14],points_artificial_court[8],points_artificial_court[7])
  elif option == 'bottom_left_serve':
    return is_inside_rectangle(land_position_real,points_artificial_court[6],points_artificial_court[5],points_artificial_court[16],points_artificial_court[15])
  elif option == 'bottom_right_serve':
    return is_inside_rectangle(land_position_real,points_artificial_court[5],points_artificial_court[4],points_artificial_court[15],points_artificial_court[14])
  elif option == 'bottom_double_half':
    return is_inside_rectangle(land_position_real,points_artificial_court[3],points_artificial_court[0],points_artificial_court[20],points_artificial_court[19])
  elif option == 'top_double_half':
    return is_inside_rectangle(land_position_real,points_artificial_court[20],points_artificial_court[19],points_artificial_court[13],points_artificial_court[10])
  elif option == 'double_court':
    return is_inside_rectangle(land_position_real,points_artificial_court[3],points_artificial_court[0],points_artificial_court[13],points_artificial_court[10])      

    
    
def is_inside_rectangle(point, left_bottom, right_bottom, left_top, right_top):
  """
    Description:
      Check if point is insidee rectangle
    Parameters:
      point                                    (int,int)          : point to check
      left_bottom                              (int,int)          : left_bottom point of rectangle
      right_bottom                             (int,int)          : right_bottom point of rectangle
      left_top                                 (int,int)          : left_top point of rectangle
      right_top                                (int,int)          : right_top point of rectangle
    Returns:
                                                bool              : True if point inside rectangle, False otherwise
  """
  point_x = point[0] 
  point_y = point[1]

  if point_x >= left_bottom[0] and point_x <= right_bottom[0] and point_y >= left_top[1] and point_y <= left_bottom[1]:
    return True
  else:
    return False    
    
    
    
def classify_event(index_of_event,events_trajectory_indexes, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court):
  """
    Description:
      Classify event by direction in y before and after event, if same direction then land, otherwise hit
    Parameters:
      index_of_event                                       int                                 : index of event to check
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      homographies_image_to_artificial_court_per_frame    list(homography)                    : list of homographies per image from cv2.findHomography()
      points_artificial_court                             list(list(tuple[int,int]))          : list of court points of artificial court with real world dimensions
    Returns:
                                                          string                              : 'land_near' or 'land_far' or 'hit_near' or 'hit_far'
  """
  
  current_trajectory_index = events_trajectory_indexes[index_of_event]
  current_frame_index = trajectory[current_trajectory_index][0]

  if len(events_trajectory_indexes) == 1:
    most_backward_index = 0
    most_forward_index = len(trajectory) -1
  elif index_of_event == 0:
    most_backward_index = max(events_trajectory_indexes[index_of_event]-3,0)#0
    most_forward_index = events_trajectory_indexes[index_of_event+1] 
  elif index_of_event == (len(events_trajectory_indexes)-1):
    most_backward_index = events_trajectory_indexes[index_of_event-1] 
    most_forward_index = len(trajectory) -1
  else:
    most_backward_index = events_trajectory_indexes[index_of_event-1] 
    most_forward_index = events_trajectory_indexes[index_of_event+1] 

  velocity_before_event_y = trajectory[current_trajectory_index][2] - trajectory[most_backward_index][2]
  velocity_after_event_y = trajectory[most_forward_index][2] - trajectory[current_trajectory_index][2]

  sign1 = velocity_before_event_y >= 0
  sign2 = velocity_after_event_y >= 0
  
  ball_position = (trajectory[current_trajectory_index][1],trajectory[current_trajectory_index][2])

  if sign1 == sign2:
    return classify_landing_spot2(current_trajectory_index,trajectory,current_frame_index,homographies_image_to_artificial_court_per_frame, points_artificial_court)
  else:
    closer_player = to_which_player_closer(ball_position, players_detections, current_frame_index)
    if closer_player == 'player_near':#
      return 'hit_near'
    elif closer_player == 'player_far':
      return 'hit_far'


def to_which_player_closer(position, players_detections, current_frame_index):
  """
    Description:
      Classify if position closer to bottom or top player
    Parameters:
      position                                            int,int                             : x,y position
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      current_frame_index                                 int                                 : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
    Returns:
                                                          string                              : 'player_near' or 'player_far' 
  """

  player_top_bb = players_detections['player_top'][current_frame_index]
  player_bottom_bb = players_detections['player_bottom'][current_frame_index]
  player_top_position = (int((player_top_bb.xmax + player_top_bb.xmin)/2),int((player_top_bb.ymax + player_top_bb.ymin)/2))
  player_bottom_position = (int((player_bottom_bb.xmax + player_bottom_bb.xmin)/2),int((player_bottom_bb.ymax + player_bottom_bb.ymin)/2))

  dist_to_top_player = compute_distance(player_top_position, position)
  dist_to_bottom_player = compute_distance(player_bottom_position, position)
  if dist_to_top_player > dist_to_bottom_player:#
    return 'player_near'
  else:
    return 'player_far'

    
def detect_event_candidates(trajectory):
  """
    Description:
      Mark detections in trajectory with big enough angles
    Parameters:
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
    Returns:
      events_trajectory_indexes                           list(tuple[int,list(int)])          : list with tuple of index to trajectory and list of angles
  """
  events_trajectory_indexes = []

  angle_threshold = 20
  angle_threshold_control = 10
  angle_low_threshold = 5
  for i in range(3,len(trajectory)-3):
    angles = []
    angles_control = []
    angles.append(get_angle2(trajectory[i-1][1:],trajectory[i][1:],trajectory[i+1][1:]))
    angles.append(get_angle2(trajectory[i-2][1:],trajectory[i][1:],trajectory[i+1][1:]))
    

    angles.append(get_angle2(trajectory[i-1][1:],trajectory[i][1:],trajectory[i+2][1:]))
    angles.append(get_angle2(trajectory[i-2][1:],trajectory[i][1:],trajectory[i+2][1:]))
    

    angles_control.append(get_angle2(trajectory[i-2][1:],trajectory[i-1][1:],trajectory[i+2][1:]))
    angles_control.append(get_angle2(trajectory[i-2][1:],trajectory[i+1][1:],trajectory[i+2][1:]))
    
    angles_control.append(get_angle2(trajectory[i-2][1:],trajectory[i-1][1:],trajectory[i+1][1:]))
    angles_control.append(get_angle2(trajectory[i-1][1:],trajectory[i+1][1:],trajectory[i+2][1:]))
    
    cnt = 0
    cnt2 = 0
    for angle in angles:
      if angle >= angle_threshold:
        cnt = cnt + 1  
      if angle >= angle_low_threshold:
        cnt2 = cnt2 + 1

    if cnt2 >= 4 and cnt >= 1 and ((angles_control[0] > angle_threshold_control or angles_control[1] > angle_threshold_control) or (angles_control[2] > angle_threshold_control or angles_control[3] > angle_threshold_control)):
      frame_num = trajectory[i][0]
      events_trajectory_indexes.append((i,angles))
      
  return events_trajectory_indexes

def is_bigger_min_angle(angles1,angles2):
  """
    Description:
      Check if angles1 has bigger minimal angle
    Parameters:
      angles1                                          list(int)            : list of angles
      angles2                                          list(int)            : list of angles
    Returns:
                                                       bool          : True if angles1 has bigger minimal angle, False otherwise
  """
  angles1 = sorted(angles1)
  angles2 = sorted(angles2)
  for i in range(len(angles1)):
    if angles1[i] < angles2[i]:
      return False
    elif angles1[i] > angles2[i]:
      return True
  return False
  
  
def filter_events_belonging_to_one_direction_shift(events_trajectory_indexes_with_angles):
  """
    Description:
      If event detection in trajectory on position i has bigger minimal angle than event detections on positions i+1, i-1, if yes delete 
    Parameters:
       events_trajectory_indexes                           list(tuple[int,list(int)])          : list with tuple of index to trajectory and list of angles
    Returns:
      events_trajectory_indexes                             list(int)                          : list with indexes of event after filtering
  """

  for i in range(len(events_trajectory_indexes_with_angles)-1,-1,-1):
    angles_current = events_trajectory_indexes_with_angles[i][1]
    delete_current = False

    for j in range(min(len(events_trajectory_indexes_with_angles)-1-i,1)): 
      if abs(events_trajectory_indexes_with_angles[i][0] - events_trajectory_indexes_with_angles[i+j+1][0]) <= 1:
        competitive_angles = events_trajectory_indexes_with_angles[i+j+1][1]
        if is_bigger_min_angle(competitive_angles,angles_current):
          delete_current = True
    for j in range(min(i,1)): 
      if abs(events_trajectory_indexes_with_angles[i][0] - events_trajectory_indexes_with_angles[i-j-1][0]) <= 1:
        competitive_angles = events_trajectory_indexes_with_angles[i-j-1][1]
        if is_bigger_min_angle(competitive_angles,angles_current):
          delete_current = True
    if delete_current:
      events_trajectory_indexes_with_angles.pop(i)

  events_trajectory_indexes = []    
  for item in events_trajectory_indexes_with_angles: 
    events_trajectory_indexes.append(item[0])
  return events_trajectory_indexes
  
  
def filter_events(indexes_to_filter,events_trajectory_indexes, events_indexes_classificated, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height):
  """
    Description:
      Filter events in indexes_to_filter
    Parameters:
      indexes_to_filter                                   list(int)                           : list of indexes of events to check
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      events_indexes_classificated  list(string)                        : list of events categories
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      homographies_image_to_artificial_court_per_frame    list(homography)                    : list of homographies per image from cv2.findHomography()
      points_artificial_court                             list(list(tuple[int,int]))          : list of court points of artificial court with real world dimensions
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
    events_trajectory_indexes                             list(int)                           : if event is removed, index is removed from list
    events_indexes_classificated                          list(string)                        : if event is removed, then is removed from list, if it is changed, then it is changed in the list
  """
  
  events_trajectory_indexes_reclassified_as_land = []
  events_trajectory_indexes_reclassified_as_hit_near = []
  events_trajectory_indexes_reclassified_as_hit_far = []
  i = 0
  while i < len(indexes_to_filter):
    index = indexes_to_filter[i]
    events_to_remove_indexes = []
    event_to_remove_index, event_trajectory_index_reclassified_as_land, event_trajectory_indexes_reclassified_as_hit_near, event_trajectory_indexes_reclassified_as_hit_far = filter_events_by_predefined_rules(index,trajectory, events_trajectory_indexes,events_indexes_classificated,players_detections, corner_points_per_images, image_width, image_height)
    if event_to_remove_index is not None:
      events_to_remove_indexes.append(event_to_remove_index)
    if event_trajectory_index_reclassified_as_land is not None:
      events_trajectory_indexes_reclassified_as_land.append(event_trajectory_index_reclassified_as_land)
    if event_trajectory_indexes_reclassified_as_hit_near is not None:
      events_trajectory_indexes_reclassified_as_hit_near.append(event_trajectory_indexes_reclassified_as_hit_near)
    if event_trajectory_indexes_reclassified_as_hit_far is not None:
      events_trajectory_indexes_reclassified_as_hit_far.append(event_trajectory_indexes_reclassified_as_hit_far)  
      

    if len(events_to_remove_indexes) >= 1:     
      events_trajectory_indexes,events_indexes_classificated = remove_events(events_trajectory_indexes, events_indexes_classificated, events_to_remove_indexes, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)
    i = i + 1 
    
  for trajectory_index in events_trajectory_indexes_reclassified_as_hit_near:
    index_of_event = events_trajectory_indexes.index(trajectory_index) #takto protoze se muze neco smazat mezi tim, proto neni primy odkaz na index_of_event
    events_indexes_classificated[index_of_event] = 'hit_near'
    
  for trajectory_index in events_trajectory_indexes_reclassified_as_hit_far:
    index_of_event = events_trajectory_indexes.index(trajectory_index) #takto protoze se muze neco smazat mezi tim, proto neni primy odkaz na index_of_event
    events_indexes_classificated[index_of_event] = 'hit_far' 
    
  for trajectory_index in events_trajectory_indexes_reclassified_as_land:
    index_of_event = events_trajectory_indexes.index(trajectory_index)
    current_trajectory_index = events_trajectory_indexes[index_of_event]
    current_frame_index = trajectory[current_trajectory_index][0]
    events_indexes_classificated[index_of_event] = classify_landing_spot2(trajectory_index, trajectory, current_frame_index, homographies_image_to_artificial_court_per_frame, points_artificial_court)
  return events_trajectory_indexes,events_indexes_classificated

  
def get_automata():
  """
    Description:
      Return automata for filtering as dictionary
    Parameters:

    Returns:
    automata                             dict{state:transitions}                           : automata for filtering
  """
  start_transition_table = {
      'land_near' : 'land_near',
      'land_far'  : 'land_far',
      'hit_far'   : 'hit_far',
      'hit_near'  : 'hit_near'
  }
  land_near_transition_table = {
      'land_near' : 'land_near',
      'land_far'  : '-1',
      'hit_far'   : 'hit_far',
      'hit_near'  : '-1'
  }
  land_far_transition_table = {
      'land_near' : '-1',
      'land_far'  : 'land_far',
      'hit_far'   : '-1',
      'hit_near'  : 'hit_near'
  }
  hit_far_transition_table = {
      'land_near' : '-1',
      'land_far'  : 'land_far',
      'hit_far'   : '-1',
      'hit_near'  : 'hit_near'
  }
  hit_near_transition_table = {
      'land_near' : 'land_near',
      'land_far'  : '-1',
      'hit_far'   : 'hit_far',
      'hit_near'  : '-1'
  }

  automata = {
      'start'           : start_transition_table,
      'land_near'       : land_near_transition_table,
      'land_far'        : land_far_transition_table,
      'hit_far'         : hit_far_transition_table,
      'hit_near'        : hit_near_transition_table,
  }
  return automata

def get_center_from_bounding_box(bounding_box):
  """
    Description:
      Return center of bounding box
    Parameters:
    bounding_box                         boundingbox                         : ...
    Returns:
                                 (int,int)                                   : center of bounding box
  """
  return ((bounding_box.xmin + bounding_box.xmax)/2,(bounding_box.ymin + bounding_box.ymax)/2)
def get_top_center_from_bounding_box(bounding_box):
  """
    Description:
      Return top center of bounding box
    Parameters:
    bounding_box                         boundingbox                         : ...
    Returns:
                                 (int,int)                                   : top center of bounding box
  """
  return ((bounding_box.xmin + bounding_box.xmax)/2,bounding_box.ymin)
  
def run_automata(start_state, automata,start_index, events_indexes_classificated):
  """
    Description:
      Run automata, if error state or end of sequence return
    Parameters:
    bounding_box                         boundingbox                         : ...
    Returns:
                                 (int,int)                                   : top center of bounding box
  """
  state = start_state
  transition = ''
  events_to_remove_indexes = None
  old_state1 = start_state
  old_state2 = 'start'
  old_state3 = 'start'
  old_state4 = 'start'
  old_state5 = 'start'
  for index in range(start_index,-1,-1):
    transition = events_indexes_classificated[index]
    state = automata[state][transition]
    if state == '-1':
      events_to_remove_indexes = index
      break
    old_state5 = old_state4
    old_state4 = old_state3
    old_state3 = old_state2
    old_state2 = old_state1
    old_state1 = state 
  return events_to_remove_indexes, old_state1, old_state2, old_state3, old_state4, old_state5 

def filter_events_by_automata(automata, events_trajectory_indexes, events_indexes_classificated, trajectory,players_detections,corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height):
  """
    Description:
      Run automata, if error state try to repair by rules
    Parameters:
      automata                                   dict{state:transitions}                           : automata for filtering
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      events_indexes_classificated  list(string)                        : list of events categories
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      homographies_image_to_artificial_court_per_frame    list(homography)                    : list of homographies per image from cv2.findHomography()
      points_artificial_court                             list(list(tuple[int,int]))          : list of court points of artificial court with real world dimensions
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
    events_trajectory_indexes                             list(int)                           : if event is removed, index is removed from list
    events_indexes_classificated                          list(string)                        : if event is removed, then is removed from list, if it is changed, then it is changed in the list
  """
  state_to_start = 'start'
  start_index = len(events_indexes_classificated)-1 
  while True:

    event_to_remove_index, old_state1, old_state2, old_state3, old_state4, old_state5 = run_automata(state_to_start, automata, start_index, events_indexes_classificated) #state je aktualni z ktereho detekovana chyba, old state jeden pred nim
    if event_to_remove_index is None:
      break
    else:
      should_remove = False

      if len(events_indexes_classificated) > 1:
        if events_indexes_classificated[event_to_remove_index] == 'hit_near' and events_indexes_classificated[event_to_remove_index + 1] == 'hit_near' \
        or (events_indexes_classificated[event_to_remove_index] == 'hit_far' and events_indexes_classificated[event_to_remove_index + 1] == 'hit_far'):

          if events_indexes_classificated[event_to_remove_index] == 'hit_near' and events_indexes_classificated[event_to_remove_index + 1] == 'hit_near' and events_indexes_classificated[event_to_remove_index - 1] == 'hit_near'\
          and can_be_landing_spot(event_to_remove_index + 1,trajectory,events_trajectory_indexes,image_width, image_height) and classify_landing_spot((trajectory[events_trajectory_indexes[event_to_remove_index + 1]][1],trajectory[events_trajectory_indexes[event_to_remove_index + 1]][2]),corner_points_per_images,trajectory[events_trajectory_indexes[event_to_remove_index + 1]][0],image_width, image_height) == 'land_far':
            events_indexes_classificated[event_to_remove_index + 1] = 'land_far'
            state_to_start = old_state2
            start_index = event_to_remove_index+1          
          
          elif events_indexes_classificated[event_to_remove_index] == 'hit_near' and events_indexes_classificated[event_to_remove_index + 1] == 'hit_near'\
          and can_be_landing_spot(event_to_remove_index,trajectory,events_trajectory_indexes,image_width, image_height) and classify_landing_spot((trajectory[events_trajectory_indexes[event_to_remove_index]][1],trajectory[events_trajectory_indexes[event_to_remove_index]][2]),corner_points_per_images,trajectory[events_trajectory_indexes[event_to_remove_index]][0],image_width, image_height) == 'land_near':
            events_indexes_classificated[event_to_remove_index] = 'land_near'
            state_to_start = old_state1
            start_index = event_to_remove_index

          elif events_indexes_classificated[event_to_remove_index] == 'hit_far' and events_indexes_classificated[event_to_remove_index + 1] == 'hit_far'\
          and can_be_landing_spot(event_to_remove_index,trajectory,events_trajectory_indexes,image_width, image_height) and classify_landing_spot((trajectory[events_trajectory_indexes[event_to_remove_index]][1],trajectory[events_trajectory_indexes[event_to_remove_index]][2]),corner_points_per_images,trajectory[events_trajectory_indexes[event_to_remove_index]][0],image_width, image_height) == 'land_far':
            events_indexes_classificated[event_to_remove_index] = 'land_far'
            state_to_start = old_state1
            start_index = event_to_remove_index

          elif events_indexes_classificated[event_to_remove_index] == 'hit_near' and events_indexes_classificated[event_to_remove_index + 1] == 'hit_near'\
          and can_be_landing_spot(event_to_remove_index+1,trajectory,events_trajectory_indexes,image_width, image_height) and classify_landing_spot((trajectory[events_trajectory_indexes[event_to_remove_index+1]][1],trajectory[events_trajectory_indexes[event_to_remove_index+1]][2]),corner_points_per_images,trajectory[events_trajectory_indexes[event_to_remove_index+1]][0],image_width, image_height) == 'land_far':
            events_indexes_classificated[event_to_remove_index+1] = 'land_far'
            state_to_start = old_state2
            start_index = event_to_remove_index+1

          elif events_indexes_classificated[event_to_remove_index] == 'hit_far' and events_indexes_classificated[event_to_remove_index + 1] == 'hit_far'\
          and can_be_landing_spot(event_to_remove_index+1,trajectory,events_trajectory_indexes,image_width, image_height) and classify_landing_spot((trajectory[events_trajectory_indexes[event_to_remove_index+1]][1],trajectory[events_trajectory_indexes[event_to_remove_index+1]][2]),corner_points_per_images,trajectory[events_trajectory_indexes[event_to_remove_index+1]][0],image_width, image_height) == 'land_near':
            events_indexes_classificated[event_to_remove_index+1] = 'land_near'
            state_to_start = old_state2
            start_index = event_to_remove_index+1
            

          else:  
            if events_indexes_classificated[event_to_remove_index] == 'hit_near':
              dist1 = get_dist_to_player(trajectory, events_trajectory_indexes[event_to_remove_index+1],players_detections, 'player_near')
              dist2 = get_dist_to_player(trajectory, events_trajectory_indexes[event_to_remove_index],players_detections, 'player_near')
            else:
              dist1 = get_dist_to_player(trajectory, events_trajectory_indexes[event_to_remove_index+1],players_detections, 'player_far')
              dist2 = get_dist_to_player(trajectory, events_trajectory_indexes[event_to_remove_index],players_detections, 'player_far')
            
            if dist1 > dist2: 
              state_to_start = old_state3
              start_index = event_to_remove_index + 1
              if (event_to_remove_index + 1) == (len(events_indexes_classificated)-1): 
                indexes_to_filter = [event_to_remove_index + 1 - 1] 
                start_index = start_index - 1
                state_to_start = 'start'
              elif (event_to_remove_index + 1) == (len(events_indexes_classificated)-2):
                indexes_to_filter = [event_to_remove_index + 1, event_to_remove_index + 1 - 1] 
                state_to_start = 'start'
              else:
                indexes_to_filter = [event_to_remove_index + 1, event_to_remove_index + 1 - 1] 
              
              events_trajectory_indexes,events_indexes_classificated = remove_events(events_trajectory_indexes, events_indexes_classificated, [event_to_remove_index+1], trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)

              size_before = len(events_trajectory_indexes)
              events_trajectory_indexes,events_indexes_classificated = filter_events(indexes_to_filter,events_trajectory_indexes, events_indexes_classificated, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)
              cnt_remove = size_before - len(events_trajectory_indexes)
              start_index = start_index - cnt_remove

            else: 
              state_to_start = old_state2
              start_index = event_to_remove_index + 0 
              
              if (event_to_remove_index + 0) == (len(events_indexes_classificated)-1): 
                indexes_to_filter = [event_to_remove_index - 1]
                start_index = start_index - 1
                state_to_start = 'start'
              elif (event_to_remove_index + 0) == (len(events_indexes_classificated)-2):
                indexes_to_filter = [event_to_remove_index, event_to_remove_index-1]
                state_to_start = 'start'
              else:
                indexes_to_filter = [event_to_remove_index, event_to_remove_index-1]
              
              events_trajectory_indexes,events_indexes_classificated = remove_events(events_trajectory_indexes, events_indexes_classificated, [event_to_remove_index], trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)

              size_before = len(events_trajectory_indexes)
              events_trajectory_indexes,events_indexes_classificated = filter_events(indexes_to_filter,events_trajectory_indexes, events_indexes_classificated, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)
              cnt_remove = size_before - len(events_trajectory_indexes)
              start_index = start_index - cnt_remove
        #land far / hit far after land near      
        elif (event_to_remove_index != 0) and (events_indexes_classificated[event_to_remove_index] == 'land_near' and events_indexes_classificated[event_to_remove_index - 1] == 'hit_near'  \
        and can_be_landing_spot(event_to_remove_index - 1,trajectory,events_trajectory_indexes,image_width, image_height)):
          events_indexes_classificated[event_to_remove_index] = 'hit_near'
          events_indexes_classificated[event_to_remove_index-1] = 'land_near'
          state_to_start = old_state1
          start_index = event_to_remove_index
        #land near / hit near after land far    
        elif (event_to_remove_index != 0) and (events_indexes_classificated[event_to_remove_index] == 'land_far' and events_indexes_classificated[event_to_remove_index - 1] == 'hit_far'  \
        and can_be_landing_spot(event_to_remove_index - 1,trajectory,events_trajectory_indexes,image_width, image_height)):
          events_indexes_classificated[event_to_remove_index] = 'hit_far'
          events_indexes_classificated[event_to_remove_index-1] = 'land_far'
          state_to_start = old_state1
          start_index = event_to_remove_index
          
        
        elif events_indexes_classificated[event_to_remove_index] == 'hit_far' and events_indexes_classificated[event_to_remove_index + 1] == 'land_far':          
          #land far after hit far should mean that no more hit near or land near far in trajectory, if they are, then land_far after hit far is wrong an will be removed
          #also no more hits far after this one, so clean them
          should_clean = True
          for i in range(event_to_remove_index+2, len(events_indexes_classificated)):
            if events_indexes_classificated[i] == 'hit_near' or events_indexes_classificated[i] == 'land_near':
              should_clean = False
              break
          
          if should_clean == True:
            to_remove = []  
            for i in range(event_to_remove_index+2, len(events_indexes_classificated)):
              if events_indexes_classificated[i] == 'hit_near' or events_indexes_classificated[i] == 'land_near':
                break
              if events_indexes_classificated[i] == 'hit_far':
                dist1 = get_dist_to_player(trajectory, events_trajectory_indexes[event_to_remove_index],players_detections, 'player_far')#compute_distance(ball_position1,player_position1)
                dist2 = get_dist_to_player(trajectory, events_trajectory_indexes[i],players_detections, 'player_far')#compute_distance(ball_position2,player_position2)
                if dist1 > dist2:
                  should_remove = True #remove this hit far, because some hit far after this one is closer to player
                  break
                else:
                  to_remove.append(i)
            for i in range(len(to_remove)-1,-1,-1):      
                events_indexes_classificated.pop(to_remove[i])
                events_trajectory_indexes.pop(to_remove[i])
          #remove hit if land far has different directions before and after event than positive and positive      
          if direction_check(trajectory, events_trajectory_indexes, event_to_remove_index+1, 'positive', 'positive') == True and should_remove == False:
            remove_land = False
            for k in range(event_to_remove_index + 1,len(events_indexes_classificated)):
              if events_indexes_classificated[k] == 'land_near' or events_indexes_classificated[k] == 'hit_near':
                remove_land = True
                
            if remove_land == True:
              should_remove = True
              size = len(events_trajectory_indexes)
              if size <=  event_to_remove_index + 3:
                old_state2 = 'start'
              else:
                old_state2 = events_indexes_classificated[event_to_remove_index + 3]
                
              event_to_remove_index = event_to_remove_index + 1  
            else:
              state_to_start = 'hit_far'
              start_index = event_to_remove_index-1
              
          else:#remove hit_far
            should_remove = True
              
        #same case as above, but for 'near'     
        elif events_indexes_classificated[event_to_remove_index] == 'hit_near' and events_indexes_classificated[event_to_remove_index + 1] == 'land_near':

          should_clean = True
          for i in range(event_to_remove_index+2, len(events_indexes_classificated)):
            if events_indexes_classificated[i] == 'hit_far' or events_indexes_classificated[i] == 'land_far':
              should_clean = False
              break
          
          if should_clean == True:
            to_remove = []  
            for i in range(event_to_remove_index+2, len(events_indexes_classificated)):
              if events_indexes_classificated[i] == 'hit_far' or events_indexes_classificated[i] == 'land_far':
                break
              if events_indexes_classificated[i] == 'hit_near':
                dist1 = get_dist_to_player(trajectory, events_trajectory_indexes[event_to_remove_index],players_detections, 'player_near')
                dist2 = get_dist_to_player(trajectory, events_trajectory_indexes[i],players_detections, 'player_near')
                if dist1 > dist2:
                  should_remove = True
                  break
                else:
                  to_remove.append(i)
            for i in range(len(to_remove)-1,-1,-1):      
                events_indexes_classificated.pop(to_remove[i])
                events_trajectory_indexes.pop(to_remove[i])

          if direction_check(trajectory, events_trajectory_indexes, event_to_remove_index+1, 'negative', 'negative') == True and should_remove == False:
            
            remove_land = False
            for k in range(event_to_remove_index + 1,len(events_indexes_classificated)):
              if events_indexes_classificated[k] == 'land_far' or events_indexes_classificated[k] == 'hit_far':
                remove_land = True
                
            if remove_land == True:
              should_remove = True
              size = len(events_trajectory_indexes)
              if size <=  event_to_remove_index + 3:
                old_state2 = 'start'
              else:
                old_state2 = events_indexes_classificated[event_to_remove_index + 3]
              event_to_remove_index = event_to_remove_index + 1 
            else:
              state_to_start = 'hit_near'
              start_index = event_to_remove_index-1  
          
          else:
            should_remove = True
   
        elif (events_indexes_classificated[event_to_remove_index] == 'land_far' and events_indexes_classificated[event_to_remove_index + 1] == 'hit_near'  \
        and can_be_landing_spot(event_to_remove_index + 1,trajectory,events_trajectory_indexes,image_width, image_height)):
          events_indexes_classificated[event_to_remove_index] = 'hit_near'
          events_indexes_classificated[event_to_remove_index+1] = 'land_far'
          state_to_start = old_state2
          start_index = event_to_remove_index+1

        elif (events_indexes_classificated[event_to_remove_index] == 'land_near' and events_indexes_classificated[event_to_remove_index + 1] == 'hit_far'  \
        and can_be_landing_spot(event_to_remove_index + 1,trajectory,events_trajectory_indexes,image_width, image_height)):
          events_indexes_classificated[event_to_remove_index] = 'hit_far'
          events_indexes_classificated[event_to_remove_index+1] = 'land_near'
          state_to_start = old_state2
          start_index = event_to_remove_index+1
          
        else:
          should_remove = True
          
          
        if should_remove == True:
          state_to_start = old_state2
          if event_to_remove_index == (len(events_trajectory_indexes)-1) or event_to_remove_index == (len(events_trajectory_indexes)-2):
            start_index = event_to_remove_index -1
            state_to_start = 'start'
          else:
              start_index = event_to_remove_index + 0 
              
          events_trajectory_indexes,events_indexes_classificated = remove_events(events_trajectory_indexes, events_indexes_classificated, [event_to_remove_index], trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)
                       
          size_before = len(events_trajectory_indexes)
          if event_to_remove_index >= size_before:
            indexes_to_filter = [event_to_remove_index-1]
          elif event_to_remove_index == 0:
            indexes_to_filter = [event_to_remove_index]
          else:
             indexes_to_filter = [event_to_remove_index, event_to_remove_index-1]
             
          events_trajectory_indexes,events_indexes_classificated = filter_events(indexes_to_filter,events_trajectory_indexes, events_indexes_classificated, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)
          cnt_remove = size_before - len(events_trajectory_indexes)
          start_index = start_index - cnt_remove
          if cnt_remove > 0:
            if start_index != (len(events_trajectory_indexes)-1) and start_index != (len(events_trajectory_indexes)-2):#len(events_indexes_classificated) > (start_index+1):
              state_to_start = events_indexes_classificated[start_index+1]
            else:
              state_to_start = 'start'
      else:
        events_trajectory_indexes = []
        events_indexes_classificated = []
  return events_trajectory_indexes, events_indexes_classificated

def get_dist_to_player(trajectory, trajectory_index,players_detections, which_player):
  """
    Description:
     Get distance from ball to specified plauer
    Parameters:
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      trajectory_index                                    int                                 : index of trajectory detection to check
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      which_player                                        string                              : 'player_near' or 'player_far'
    Returns:
                                                          float                               : distance from ball to player
  """
  current_frame_index = trajectory[trajectory_index][0]
  ball_position = (trajectory[trajectory_index][1], trajectory[trajectory_index][2])

  if which_player == 'player_near':
    player_position = get_center_from_bounding_box(players_detections['player_bottom'][current_frame_index])
  elif which_player == 'player_far':
    player_position = get_center_from_bounding_box(players_detections['player_top'][current_frame_index])
      
  return compute_distance(ball_position,player_position)


  
def direction_check(trajectory, events_trajectory_indexes, index_of_event, dir1, dir2):
  """
    Description:
     Check if direction before and after event match dri1 and dir2
    Parameters:
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      index_of_event                                      int                                 : index of event to check
      dir1                                                string                              : 'negative' or 'positive'
      dir2                                                string                              : 'negative' or 'positive'
    Returns:
                                                          bool                                : True if directions matches, False otherwise
  """
  current_trajectory_index = events_trajectory_indexes[index_of_event]
  if len(events_trajectory_indexes) == 1:
    most_backward_index = 0
    most_forward_index = len(trajectory) -1
  elif index_of_event == 0:
    most_backward_index = 0
    most_forward_index = events_trajectory_indexes[index_of_event+1] 
  elif index_of_event == (len(events_trajectory_indexes)-1):
    most_backward_index = events_trajectory_indexes[index_of_event-1] 
    most_forward_index = len(trajectory) -1
  else:
    most_backward_index = events_trajectory_indexes[index_of_event-1] 
    most_forward_index = events_trajectory_indexes[index_of_event+1] 

  velocity_before_event_y = trajectory[current_trajectory_index][2] - trajectory[most_backward_index][2]
  velocity_after_event_y = trajectory[most_forward_index][2] - trajectory[current_trajectory_index][2]
  
  if dir1 == 'negative' and dir2 == 'negative':
    return velocity_before_event_y < 0 and velocity_after_event_y < 0
  elif dir1 == 'negative' and dir2 == 'positive':
    return velocity_before_event_y < 0 and velocity_after_event_y > 0
  elif dir1 == 'positive' and dir2 == 'negative':
    return velocity_before_event_y > 0 and velocity_after_event_y < 0
  elif dir1 == 'positive' and dir2 == 'positive':
    return velocity_before_event_y > 0 and velocity_after_event_y > 0




def where_player_stands(current_frame_index, players_detections, corner_points_per_images, player_near_or_far):
  """
    Description:
      Decide where serving player stands, left or right
    Parameters:
      current_frame_index                                 int                                 : index of frame where to check
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      player_near_or_far                                  string                              : check near or far player, values 'player_far' or 'player_near'
    Returns:
                                                          string                              : where player stands, values 'player_far_right' or 'player_far_left' or 'player_near_right' or 'player_near_left'
  """
  
  if player_near_or_far == 'player_far':
    top_baseline_half_point_x = corner_points_per_images[current_frame_index][18][0]
    top_player = players_detections['player_top'][current_frame_index]
    top_player_x = (top_player.xmax + top_player.xmin) / 2
    if top_baseline_half_point_x > top_player_x:
      return 'player_far_right'
    else:
      return 'player_far_left'
  elif player_near_or_far == 'player_near':
    bottom_baseline_half_point_x = corner_points_per_images[current_frame_index][17][0]
    bottom_player = players_detections['player_bottom'][current_frame_index]
    bottom_player_x = (bottom_player.xmax + bottom_player.xmin) / 2
    if bottom_baseline_half_point_x > bottom_player_x:
      return 'player_near_left'
    else:
      return 'player_near_right'
  return None

def which_player_where_serve(trajectory, events_indexes_classificated, events_trajectory_indexes, players_detections, corner_points_per_images, image_width, image_height):
  """
    Description:
      Decide which player serve
    Parameters:
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      events_indexes_classificated                        list(string)                        : list of events categories
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
                                                          string,int                          : whcih player where serve and index of event of ball landing after serve
  """


  for i in range(len(events_indexes_classificated)):
    current_trajectory_index = events_trajectory_indexes[i]
    current_frame_index = trajectory[current_trajectory_index][0]
    ball_position = (trajectory[current_trajectory_index][1],trajectory[current_trajectory_index][2])
    bottom_right_corner =       corner_points_per_images[current_frame_index][0]
    bottom_left_corner =        corner_points_per_images[current_frame_index][3]
    top_right_corner =          corner_points_per_images[current_frame_index][10]
    top_left_corner =           corner_points_per_images[current_frame_index][13]
    serve_middle_right_corner =   corner_points_per_images[current_frame_index][19]
    serve_middle_left_corner =    corner_points_per_images[current_frame_index][20]
    
    if events_indexes_classificated[i] == 'land_far' \
    and is_inside_area(ball_position, serve_middle_right_corner, serve_middle_left_corner, top_right_corner, top_left_corner, image_width, image_height) == 'in':

      is_ball_over_net = False
      for j in range(current_trajectory_index,-1,-1):
        ball_position = (trajectory[j][1],trajectory[j][2])
        current_frame_index = trajectory[j][0]
        left_corner_base_line_top_double = corner_points_per_images[current_frame_index][13]
        right_corner_base_line_top_double = corner_points_per_images[current_frame_index][10]
        half_base_line_top = corner_points_per_images[current_frame_index][18]
        right_net_end = corner_points_per_images[current_frame_index][23]
        left_net_end = corner_points_per_images[current_frame_index][21]
        middle_net = corner_points_per_images[current_frame_index][22]
        #half line

        ball_in_left_area1 = is_inside_area(ball_position, middle_net, left_net_end, half_base_line_top, left_corner_base_line_top_double, image_width, image_height)
        ball_in_right_area1 = is_inside_area(ball_position, right_net_end, middle_net, right_corner_base_line_top_double, half_base_line_top, image_width, image_height)
        ball_position = (ball_position[0], ball_position[1] - 10) ##tolerance due to imperfection of detection, move little bit down
        ball_in_left_area2 = is_inside_area(ball_position, middle_net, left_net_end, half_base_line_top, left_corner_base_line_top_double, image_width, image_height)
        ball_in_right_area2 = is_inside_area(ball_position, right_net_end, middle_net, right_corner_base_line_top_double, half_base_line_top, image_width, image_height)
        
        if ball_in_left_area1 == 'in' or ball_in_left_area2 == 'in' or ball_in_right_area1 == 'in' or ball_in_right_area2 == 'in': #micek prekrocil pasku
          is_ball_over_net = True
          break
      
      
      #first detection closer to net than to top baseline
      ball_position = (trajectory[0][1],trajectory[0][2])
      current_frame_index = trajectory[0][0]
      left_corner_base_line_top_double = corner_points_per_images[current_frame_index][13]
      right_corner_base_line_top_double = corner_points_per_images[current_frame_index][10]
      right_net_end = corner_points_per_images[current_frame_index][23]
      left_net_end = corner_points_per_images[current_frame_index][21]
      middle_net = corner_points_per_images[current_frame_index][22]
      
      dist_to_top_baseline = get_distance_point_line(ball_position, left_corner_base_line_top_double, right_corner_base_line_top_double)
      dist_to_net1 = get_distance_point_line(ball_position, right_net_end, middle_net)
      dist_to_net2 = get_distance_point_line(ball_position, left_net_end, middle_net)
      
      #serve hit as land far, toss mode should be previous event
      hit_as_land = True
      if i != 0:
        serve_land_position_y = trajectory[events_trajectory_indexes[i]][2]
        previous_event_index = events_trajectory_indexes[i-1]
        serve_land_index = events_trajectory_indexes[i]
        for j in range(serve_land_index-1, previous_event_index-1, -1):
          ball_position_y = trajectory[j][2]
          if (serve_land_position_y - ball_position_y) < 0:#serve land je pod ball
            hit_as_land = False
            break
      else:
        hit_as_land = False
      
      if is_ball_over_net and ((dist_to_net1 < dist_to_top_baseline) or (dist_to_net2 < dist_to_top_baseline)) and (hit_as_land == False):#or protoze kdyz nadhoz tesne nad sit, tak zalezi kde je jaka net
        return where_player_stands(current_frame_index, players_detections, corner_points_per_images, 'player_near'),i
      
      
      
    elif events_indexes_classificated[i] == 'land_near'\
    and is_inside_area(ball_position, bottom_right_corner, bottom_left_corner, serve_middle_right_corner, serve_middle_left_corner, image_width, image_height) == 'in':
      is_ball_below_half = False
      is_ball_over_net = False
      #is below half test
      for j in range(current_trajectory_index,-1,-1):
        ball_position = (trajectory[j][1],trajectory[j][2])
        frame_index = trajectory[j][0]
        left_corner_base_line_bottom_double = corner_points_per_images[frame_index][3]
        right_corner_base_line_bottom_double = corner_points_per_images[frame_index][0]
        #half line
        right_half_double_line = corner_points_per_images[frame_index][19]
        left_half_double_line = corner_points_per_images[frame_index][20]
        
        ball_in_area_below_half1 = is_inside_area(ball_position, right_corner_base_line_bottom_double, left_corner_base_line_bottom_double, right_half_double_line, left_half_double_line, image_width, image_height)
        if ball_in_area_below_half1 == 'in': 
          is_ball_below_half = True
       
        left_corner_base_line_top_double = corner_points_per_images[current_frame_index][13]
        right_corner_base_line_top_double = corner_points_per_images[current_frame_index][10]
        half_base_line_top = corner_points_per_images[current_frame_index][18]
        right_net_end = corner_points_per_images[current_frame_index][23]
        left_net_end = corner_points_per_images[current_frame_index][21]
        middle_net = corner_points_per_images[current_frame_index][22]
        ball_in_left_area1 = is_inside_area(ball_position, middle_net, left_net_end, half_base_line_top, left_corner_base_line_top_double, image_width, image_height)
        ball_in_right_area1 = is_inside_area(ball_position, right_net_end, middle_net, right_corner_base_line_top_double, half_base_line_top, image_width, image_height)
        

        if ball_in_left_area1 == 'in' or  ball_in_right_area1 == 'in': #ball crossed net
          is_ball_over_net = True
       
      #dist to top baseline is shorter than to line of first detection
      ball_position = (trajectory[0][1],trajectory[0][2])
      current_frame_index = trajectory[0][0]
      left_corner_base_line_top_double = corner_points_per_images[current_frame_index][13]
      right_corner_base_line_top_double = corner_points_per_images[current_frame_index][10]
      right_net_end = corner_points_per_images[current_frame_index][23]
      left_net_end = corner_points_per_images[current_frame_index][21]
      middle_net = corner_points_per_images[current_frame_index][22]   
      
      dist_to_top_baseline = get_distance_point_line(ball_position, left_corner_base_line_top_double, right_corner_base_line_top_double)
      dist_to_net1 = get_distance_point_line(ball_position, right_net_end, middle_net)
      dist_to_net2 = get_distance_point_line(ball_position, left_net_end, middle_net)  
          
      
      if is_ball_below_half and is_ball_over_net and ((dist_to_net1 > dist_to_top_baseline) and (dist_to_net2 > dist_to_top_baseline)):
        return where_player_stands(current_frame_index, players_detections, corner_points_per_images, 'player_far'),i
        
  return None, None
  

def find_land_serve_event(events_indexes_classificated,events_trajectory_indexes,trajectory,players_detections,corner_points_per_images, tenis_game_ending, image_width, image_height):
  """
    Description:
      Find landing of ball after serve
    Parameters:
      events_indexes_classificated                        list(string)                        : list of events categories
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      tenis_game_ending                                   tuple(int,string) or string         : index and type of game ending, checks if it is land to net, if not found then value is 'Undecided'
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
    events_trajectory_indexes                             list(int)                           : if event is removed, index is removed from list
    events_indexes_classificated                          list(string)                        : if event is removed, then is removed from list, if it is changed, then it is changed in the list
    serve_land_trajectory_index                           int or None                         : trajectory index of serve land, None if not found
    event                                                 string or None                      : serving player and side if found, None if not found
    tenis_game_ending                                 string                              : changed to serve to net if found
    serve_land_event_index                                int or None                         : event index of serve landing, None if not found
  """
  
  serve_land_event_index = None
  is_serve_to_net = False
  if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF':
    is_serve_to_net = True
    for i in range(len(events_indexes_classificated)-1):
      if events_indexes_classificated[i] == 'land_far' and events_indexes_classificated[i+1] == 'hit_far':
        is_serve_to_net = False
      elif events_indexes_classificated[i] == 'land_near' and events_indexes_classificated[i+1] == 'hit_near':
        is_serve_to_net = False

    ball_position = (trajectory[0][1],trajectory[0][2])#first ball detection in trajectory
    current_frame_index = trajectory[0][0]
    left_corner_base_line_top_double = corner_points_per_images[current_frame_index][13]
    right_corner_base_line_top_double = corner_points_per_images[current_frame_index][10]
    right_net_end = corner_points_per_images[current_frame_index][23]
    left_net_end = corner_points_per_images[current_frame_index][21]
    middle_net = corner_points_per_images[current_frame_index][22]   
    
    dist_to_top_baseline = get_distance_point_line(ball_position, left_corner_base_line_top_double, right_corner_base_line_top_double)
    dist_to_net1 = get_distance_point_line(ball_position, right_net_end, middle_net)
    dist_to_net2 = get_distance_point_line(ball_position, left_net_end, middle_net)    
    if tenis_game_ending[1] == 'LNN' and not((dist_to_net1 < dist_to_top_baseline) or (dist_to_net2 < dist_to_top_baseline)):
      is_serve_to_net = False
    if tenis_game_ending[1] == 'LNF' and not((dist_to_net1 > dist_to_top_baseline) and (dist_to_net2 > dist_to_top_baseline)):  
      is_serve_to_net = False
      
  if is_serve_to_net == True:
    if tenis_game_ending[1] == 'LNN':#'hit_net_by_player_near':
      events_indexes_classificated = ['LNN']
      events_trajectory_indexes = [tenis_game_ending[0]]
      serve_land_trajectory_index = tenis_game_ending[0]
      current_frame_index = trajectory[0][0]
      player_position = where_player_stands(current_frame_index, players_detections, corner_points_per_images, 'player_near')
      if player_position == 'player_near_right':
        tenis_game_ending = (tenis_game_ending[0], 'serve_to_net_by_near_player_right')
        event = 'SNR'
      elif player_position == 'player_near_left':
        tenis_game_ending = (tenis_game_ending[0], 'serve_to_net_by_near_player_left')
        event = 'SNL'
        
    elif tenis_game_ending[1] == 'LNF':#'hit_net_by_player_far':
      events_indexes_classificated = ['LNF']
      events_trajectory_indexes = [tenis_game_ending[0]]
      serve_land_trajectory_index = tenis_game_ending[0]
      current_frame_index = trajectory[0][0]
      player_position = where_player_stands(current_frame_index, players_detections, corner_points_per_images, 'player_far')
      if player_position == 'player_far_right':
        tenis_game_ending = (tenis_game_ending[0], 'serve_to_net_by_far_player_right')
        event = 'SFR'
      elif player_position == 'player_far_left':
        tenis_game_ending = (tenis_game_ending[0], 'serve_to_net_by_far_player_left')
        event = 'SFL'
        
    serve_land_event_index = 0
  else:
    serving_player, serve_land_event_index = which_player_where_serve(trajectory, events_indexes_classificated, events_trajectory_indexes, players_detections, corner_points_per_images, image_width, image_height)
    event = ''
  
    if (serving_player == 'player_far_left'):
      event = 'SFL'#'serve_far_from_left'
      landing_event_to_find = 'land_near'
      events_indexes_classificated[serve_land_event_index] = 'serve_land_near_left'
    elif (serving_player == 'player_far_right'):
      event ='SFR'# 'serve_far_from_right'
      landing_event_to_find = 'land_near'
      events_indexes_classificated[serve_land_event_index] = 'serve_land_near_right'
    elif (serving_player == 'player_near_left'):
      event = 'SNL'#'serve_near_from_left'
      landing_event_to_find = 'land_far'
      events_indexes_classificated[serve_land_event_index] = 'serve_land_far_left'
    elif (serving_player == 'player_near_right'):
      event = 'SNR'#'serve_near_from_right'
      landing_event_to_find = 'land_far'
      events_indexes_classificated[serve_land_event_index] = 'serve_land_far_right'
    else:
      #ball landing after serve not found, try it is not undetected LNF
      is_LNF = True
      for i in range(len(trajectory)):
        ball_position = (trajectory[i][1],trajectory[i][2])
        current_frame_index = trajectory[i][0]
        in_or_out1 = classify_landing_spot(ball_position,corner_points_per_images,current_frame_index,image_width, image_height) #jestli je za nebo pred pulkou
        ball_position = (trajectory[i][1],trajectory[i][2] - 10)
        in_or_out2 = classify_landing_spot(ball_position,corner_points_per_images,current_frame_index,image_width, image_height)
        if in_or_out1 != 'land_far' and in_or_out2 != 'land_far':
          is_LNF = False
          break
      
      if is_LNF == True:
        serving_player = where_player_stands(trajectory[0][0], players_detections, corner_points_per_images, 'player_far')
        for i in range(len(trajectory)):
          ball_position = (trajectory[i][1],trajectory[i][2])
          current_frame_index = trajectory[i][0]
          right_half_double_line = corner_points_per_images[current_frame_index][19]
          left_half_double_line = corner_points_per_images[current_frame_index][20]
          right_net_end = corner_points_per_images[current_frame_index][23]
          left_net_end = corner_points_per_images[current_frame_index][21]
          event_in_net_area = is_inside_area(ball_position, right_half_double_line, left_half_double_line, right_net_end, left_net_end, image_width, image_height)
          if event_in_net_area == 'in':
            serve_land_trajectory_index = i
            break

        
        if (serving_player == 'player_far_left'):
          event = 'SFL'#'serve_far_from_left'
          landing_event_reclassify = 'LNF'
          tenis_game_ending = (serve_land_trajectory_index, 'serve_to_net_by_far_player_left')
        elif (serving_player == 'player_far_right'):
          event ='SFR'# 'serve_far_from_right'
          landing_event_reclassify = 'LNF'
          tenis_game_ending = (serve_land_trajectory_index, 'serve_to_net_by_far_player_right')
          
        events_indexes_classificated = [landing_event_reclassify]
        events_trajectory_indexes = [serve_land_trajectory_index]
        serve_land_event_index = 0  
      
      else:
        return events_indexes_classificated,events_trajectory_indexes,None,None,tenis_game_ending,None #nenalezeno podani
      

    serve_land_trajectory_index = events_trajectory_indexes[serve_land_event_index]

  return events_indexes_classificated,events_trajectory_indexes,serve_land_trajectory_index,event,tenis_game_ending, serve_land_event_index

def sort_by_angles(item1, item2):
  return min(item1[1]) - min(item2[1])
def sort_by_dist_to_player(item1, item2):
  return item1[2] - item2[2]
def sort_by_velocity_change(item1, item2):
  return item1[3] - item2[3]
def sort_by_velocity(item1, item2):
  return item1[4] - item2[4]
  
  
  
def find_serve_event_from_landing_spot(serve_land_trajectory_index, event, events_indexes_classificated,events_trajectory_indexes,trajectory,players_detections,corner_points_per_images, tenis_game_ending, image_width, image_height):
  """
    Description:
      Run automata, if error state try to repair by rules
    Parameters:
      serve_land_trajectory_index                         int                                 : trajectory index of serve landing
      event                                               string                              : type of serve, 'SNR','SNL','SFR' or 'SFL'
      events_indexes_classificated                        list(string)                        : list of events categories
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      tenis_game_ending                                   tuple(int,string) or string         : index and type of game ending, checks if it is land to net, if not found then value is 'Undecided'
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
    serve_trajectory_index                                int or None                         : trajectory index of serve rocket impact, if not found None returned
  """


  serve_trajectory_index = None
  angles_and_velocity_change_per_detection = []
  score = []
  for i in range(serve_land_trajectory_index-1,1,-1):
    angles = []
    angles.append(get_angle2(trajectory[i-1][1:],trajectory[i][1:],trajectory[i+1][1:]))
    angles.append(get_angle2(trajectory[i-2][1:],trajectory[i][1:],trajectory[i+1][1:]))
    angles.append(get_angle2(trajectory[i-1][1:],trajectory[i][1:],trajectory[i+2][1:]))
    if event == 'SNL' or event == 'SNR':#y coords negative direction desired, but transform so positive direction desired, so same for serve from both sides of court
      player_position = get_top_center_from_bounding_box(players_detections['player_bottom'][trajectory[i][0]])
      ball_position = (trajectory[i][1],trajectory[i][2])
      dist_to_player = compute_distance(ball_position,player_position)
      frame_dist = trajectory[i+1][0] - trajectory[i][0]
      velocity_change_current = -(trajectory[i+1][2] - trajectory[i][2])  / frame_dist
      frame_dist = trajectory[i][0] - trajectory[i-1][0]
      velocity_change_one_before = -(trajectory[i][2] - trajectory[i-1][2])   / frame_dist
    elif event == 'SFL' or event == 'SFR': #ypsilon coord positive direction desired
      player_position = get_top_center_from_bounding_box(players_detections['player_top'][trajectory[i][0]])
      ball_position = (trajectory[i][1],trajectory[i][2])
      dist_to_player = compute_distance(ball_position,player_position)
      frame_dist = trajectory[i+1][0] - trajectory[i][0]
      velocity_change_current = trajectory[i+1][2] - trajectory[i][2]  / frame_dist
      frame_dist = trajectory[i][0] - trajectory[i-1][0]
      velocity_change_one_before = trajectory[i][2] - trajectory[i-1][2]   / frame_dist

    if velocity_change_one_before == 0:
      velocity_change_one_before = 1
    velocity_change = (velocity_change_current - velocity_change_one_before) / abs(velocity_change_one_before) 
    angles_and_velocity_change_per_detection.append((i,angles,dist_to_player,velocity_change,velocity_change_current))
    score.append(0)
    
  score1 = copy.deepcopy(score)
  score2 = copy.deepcopy(score)
  
  angles_and_velocity_change_per_detection = sorted(angles_and_velocity_change_per_detection, key=cmp_to_key(sort_by_angles), reverse = True)

  for i in range(len(angles_and_velocity_change_per_detection)):
    score[(serve_land_trajectory_index-1) - angles_and_velocity_change_per_detection[i][0]] = score[(serve_land_trajectory_index-1) - angles_and_velocity_change_per_detection[i][0]] + i 
  
  if event == 'SFL' or event == 'SFR':
    angles_and_velocity_change_per_detection = sorted(angles_and_velocity_change_per_detection, key=cmp_to_key(sort_by_dist_to_player), reverse = False)
    for i in range(len(angles_and_velocity_change_per_detection)):
      score[(serve_land_trajectory_index-1) - angles_and_velocity_change_per_detection[i][0]] = score[(serve_land_trajectory_index-1) - angles_and_velocity_change_per_detection[i][0]] + i 
   
  angles_and_velocity_change_per_detection = sorted(angles_and_velocity_change_per_detection, key=cmp_to_key(sort_by_velocity_change), reverse = True)
  for i in range(len(angles_and_velocity_change_per_detection)):
    score1[(serve_land_trajectory_index-1) - angles_and_velocity_change_per_detection[i][0]] = score1[(serve_land_trajectory_index-1) - angles_and_velocity_change_per_detection[i][0]] + i
  
  angles_and_velocity_change_per_detection = sorted(angles_and_velocity_change_per_detection, key=cmp_to_key(sort_by_velocity), reverse = True)
  for i in range(len(angles_and_velocity_change_per_detection)):
    score2[(serve_land_trajectory_index-1) - angles_and_velocity_change_per_detection[i][0]] = score2[(serve_land_trajectory_index-1) - angles_and_velocity_change_per_detection[i][0]] + i 
    
  if event == 'SFL' or event == 'SFR':
    for i in range(len(score)):
      score[i] = score[i] + max(score1[i],score2[i])
  else:
    for i in range(len(score)):
      score[i] = score[i] + score1[i]
  
  serve_trajectory_index =  (serve_land_trajectory_index-1) - score.index(min(score))
  
  return serve_trajectory_index
  
  
  
  
def include_serve_to_events(serve_trajectory_index,event, events_indexes_classificated,events_trajectory_indexes):
  """
    Description:
      Include serve hit to events
    Parameters:
      serve_trajectory_index                              int                                 : index of serve hit in trajectory
      events_indexes_classificated                        list(string)                        : list of events categories
      event                                               string                              : type of serve event
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
    Returns:
    events_trajectory_indexes                             list(int)                           : if event is removed, index is removed from list
    events_indexes_classificated                          list(string)                        : if event is removed, then is removed from list, if it is changed, then it is changed in the list
  """
  for i in range(len(events_indexes_classificated)-1,-1,-1):
    if events_trajectory_indexes[i] <= serve_trajectory_index:
      events_trajectory_indexes.pop(i)
      events_indexes_classificated.pop(i)
  events_indexes_classificated.insert(0,event)
  events_trajectory_indexes.insert(0,serve_trajectory_index)
  return events_indexes_classificated,events_trajectory_indexes

  
def make_higher_anotations(events_indexes_classificated,events_trajectory_indexes, trajectory, players_detections, corner_points_per_images, image_width, image_height, homographies_image_to_artificial_court_per_frame,points_artificial_court, single_game=True):#index_of_event,events_trajectory_indexes, trajectory, players_detections, bottom_right_corner, bottom_left_corner, top_right_corner, top_left_corner, image_width, image_height):
  """
    Description:
      Classify events to higher categories
    Parameters:
      events_indexes_classificated                        list(string)                        : list of events categories
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
      homographies_image_to_artificial_court_per_frame    list(homography)                    : list of homographies per image from cv2.findHomography()
      points_artificial_court                             list(list(tuple[int,int]))          : list of court points of artificial court with real world dimensions
      single_game                                         bool                                : True if check ball landing for single game, False for double game
    Returns:
    events_indexes_classificated_new                      list(string)                        : events with new categories
  """
  
  events_indexes_classificated_new = copy.deepcopy(events_indexes_classificated)
  for i in range(len(events_indexes_classificated)-1,-1,-1):
    event = events_indexes_classificated[i]
    current_trajectory_index = events_trajectory_indexes[i]
    current_frame_index = trajectory[current_trajectory_index][0]
    ball_position = (trajectory[current_trajectory_index][1],trajectory[current_trajectory_index][2])

    if single_game:
      bottom_right_corner =       corner_points_per_images[current_frame_index][1]
      bottom_left_corner =        corner_points_per_images[current_frame_index][2]
      top_right_corner =          corner_points_per_images[current_frame_index][11]
      top_left_corner =           corner_points_per_images[current_frame_index][12]
    else:
      bottom_right_corner =       corner_points_per_images[current_frame_index][0]
      bottom_left_corner =        corner_points_per_images[current_frame_index][3]
      top_right_corner =          corner_points_per_images[current_frame_index][10]
      top_left_corner =           corner_points_per_images[current_frame_index][13]

    #points for serve
    serve_bottom_right_corner =   corner_points_per_images[current_frame_index][4]
    serve_bottom_middle_corner =  corner_points_per_images[current_frame_index][5]
    serve_bottom_left_corner =    corner_points_per_images[current_frame_index][6]

    serve_top_right_corner =      corner_points_per_images[current_frame_index][7]
    serve_top_middle_corner =     corner_points_per_images[current_frame_index][8]
    serve_top_left_corner =       corner_points_per_images[current_frame_index][9]

    serve_middle_right_corner =   corner_points_per_images[current_frame_index][14]
    serve_middle_middle_corner =  corner_points_per_images[current_frame_index][15]
    serve_middle_left_corner =    corner_points_per_images[current_frame_index][16]
      
      

    if event == 'land_far':
      in_or_out = is_inside_area(ball_position, serve_middle_right_corner, serve_middle_left_corner, top_right_corner, top_left_corner, image_width, image_height)
      if in_or_out == 'in':
        events_indexes_classificated_new[i] = 'LFI'#'land_far_in'
      elif in_or_out == 'out':
        events_indexes_classificated_new[i] = 'LFO'#'land_far_out'

    elif event == 'land_near':
      in_or_out = is_inside_area(ball_position, bottom_right_corner, bottom_left_corner, serve_middle_right_corner, serve_middle_left_corner, image_width, image_height)
      if in_or_out == 'in':
        events_indexes_classificated_new[i] = 'LNI'#'land_near_in'
      elif in_or_out == 'out':
        events_indexes_classificated_new[i] = 'LNO'#land_near_out'

    elif event == 'hit_far':
      player_top_position = players_detections['player_top'][current_frame_index]
      player_top_position_center_x_of_bb = (player_top_position.xmax + player_top_position.xmin) / 2
      if ball_position[0] > player_top_position_center_x_of_bb:#hit left and right from view of top player
        events_indexes_classificated_new[i] = 'HFL'#'hit_far_left'
      else:
        events_indexes_classificated_new[i] = 'HFR'#'hit_far_right'


    elif event == 'hit_near':
      player_bottom_position = players_detections['player_bottom'][current_frame_index]
      player_bottom_position_center_x_of_bb = (player_bottom_position.xmax + player_bottom_position.xmin) / 2
      if ball_position[0] > player_bottom_position_center_x_of_bb:#micek ej vic vpravo od hrace dole, ma ho po pravici
        events_indexes_classificated_new[i] = 'HNR'#'hit_near_right'
      else:
        #hit far left
        events_indexes_classificated_new[i] = 'HNL'#'hit_near_left'
    
    elif event == 'serve_land_near_right':
      in_or_out = is_inside_area(ball_position, serve_bottom_right_corner, serve_bottom_middle_corner, serve_middle_right_corner, serve_middle_middle_corner, image_width, image_height)
      if in_or_out == 'in':
        events_indexes_classificated_new[i] = 'SLNI'#'land_near_in'
      elif in_or_out == 'out':
        events_indexes_classificated_new[i] = 'SLNO'#'land_near_out'

    elif event == 'serve_land_near_left':
      in_or_out = is_inside_area(ball_position, serve_bottom_middle_corner, serve_bottom_left_corner, serve_middle_middle_corner, serve_middle_left_corner, image_width, image_height)
      if in_or_out == 'in':
        events_indexes_classificated_new[i] = 'SLNI'#'land_near_in'
      elif in_or_out == 'out':
        events_indexes_classificated_new[i] = 'SLNO'#'land_near_out'
        
    elif event == 'serve_land_far_right': #right, but from view of far player
      in_or_out = is_inside_area(ball_position, serve_middle_middle_corner, serve_middle_left_corner, serve_top_middle_corner, serve_top_left_corner, image_width, image_height)
      if in_or_out == 'in':
        events_indexes_classificated_new[i] = 'SLFI'#'land_far_in'
      elif in_or_out == 'out':
        events_indexes_classificated_new[i] = 'SLFO'#'land_far_out'

    elif event == 'serve_land_far_left':#left, but from view of far player
      in_or_out = is_inside_area(ball_position, serve_middle_right_corner, serve_middle_middle_corner, serve_top_right_corner, serve_top_middle_corner, image_width, image_height)
      if in_or_out == 'in':
        events_indexes_classificated_new[i] = 'SLFI'#'land_far_in'
      elif in_or_out == 'out':
        events_indexes_classificated_new[i] = 'SLFO'#'land_far_out'
        
    
  return events_indexes_classificated_new

def find_land_to_net(index,trajectory,events_trajectory_indexes, corner_points_per_images, players_detections, image_width, image_height):
  """
    Description:
      Find event of ball landing to net
    Parameters:
      index                                   		        int                                 : trajectory index to check
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y      
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
    events_trajectory_indexes                             list(int)                           : if event is removed, index is removed from list
    events_indexes_classificated                          list(string)                        : if event is removed, then is removed from list, if it is changed, then it is changed in the list
  """
  current_index = events_trajectory_indexes[index]
  current_frame_index = trajectory[current_index][0]
  right_half_double_line = corner_points_per_images[current_frame_index][19]
  left_half_double_line = corner_points_per_images[current_frame_index][20]
  right_net_end = corner_points_per_images[current_frame_index][23]
  left_net_end = corner_points_per_images[current_frame_index][21]
  

  hit_net_event = False
 
  if len(events_trajectory_indexes) == 1:
    most_backward_index = 0
    most_forward_index = len(trajectory) -1
  elif index == 0:
    most_backward_index = 0
    most_forward_index = events_trajectory_indexes[index+1] 
  elif index == (len(events_trajectory_indexes)-1):
    most_backward_index = events_trajectory_indexes[index-1] 
    most_forward_index = len(trajectory) -1
  else:
    most_backward_index = events_trajectory_indexes[index-1] 
    most_forward_index = events_trajectory_indexes[index+1] 

  velocity_before_event_y = trajectory[current_index][2] - trajectory[most_backward_index][2]
  velocity_after_event_y = trajectory[most_forward_index][2] - trajectory[current_index][2]
  
  sign1 = velocity_before_event_y >= 0
  sign2 = velocity_after_event_y > 0
 #hit to net from top player check
  if sign1 == True and sign2 == True: 
    ball_position = (trajectory[current_index][1],trajectory[current_index][2])
    event_in_net_area1 = is_inside_area(ball_position, right_half_double_line, left_half_double_line, right_net_end, left_net_end, image_width, image_height)
    ball_position = (ball_position[0], ball_position[1] + 10) #tolerance due to imperfection of detection, move little bit down, move up not necessary, event at floor not probable
    event_in_net_area2 = is_inside_area(ball_position, right_half_double_line, left_half_double_line, right_net_end, left_net_end, image_width, image_height)
    if event_in_net_area1 == 'in' or event_in_net_area2 == 'in':
      
      not_from_serve_toss_of_near_player_detected = False 
      
      from_serve_toss_of_near_player = False
      if not_from_serve_toss_of_near_player_detected == False: 
        for i in range(current_index, most_backward_index-1,-1):
          ball_position = (trajectory[i][1],trajectory[i][2])
          frame_index = trajectory[i][0]
          left_corner_base_line_bottom_double = corner_points_per_images[frame_index][3]
          right_corner_base_line_bottom_double = corner_points_per_images[frame_index][0]
          #half line
          right_half_double_line = corner_points_per_images[frame_index][19]
          left_half_double_line = corner_points_per_images[frame_index][20]
          ball_in_area_below_half1 = is_inside_area(ball_position, right_corner_base_line_bottom_double, left_corner_base_line_bottom_double, right_half_double_line, left_half_double_line, image_width, image_height)
          if ball_in_area_below_half1 == 'in': #ball in bottom half of court, cannot be from far player
            from_serve_toss_of_near_player = True
            break
      if not_from_serve_toss_of_near_player_detected == False:
        if from_serve_toss_of_near_player == True:
          return False

      ball_in_area_below_half_of_the_court = False
      for i in range(most_backward_index, len(trajectory)):
        ball_position = (trajectory[i][1],trajectory[i][2])
        #bottom base line
        left_corner_base_line_bottom_double = corner_points_per_images[current_frame_index][3]
        right_corner_base_line_bottom_double = corner_points_per_images[current_frame_index][0]
        #half line
        right_half_double_line = corner_points_per_images[current_frame_index][19]
        left_half_double_line = corner_points_per_images[current_frame_index][20]
        ball_in_area_below_half1 = is_inside_area(ball_position, right_corner_base_line_bottom_double, left_corner_base_line_bottom_double, right_half_double_line, left_half_double_line, image_width, image_height)
        ball_position = (ball_position[0], ball_position[1] - 10)#tolerance due to imperfection of detection, move little bit up
        ball_in_area_below_half2  = is_inside_area(ball_position, right_corner_base_line_bottom_double, left_corner_base_line_bottom_double, right_half_double_line, left_half_double_line, image_width, image_height)
        if ball_in_area_below_half1 == 'in' and ball_in_area_below_half2 == 'in': 
          ball_in_area_below_half_of_the_court = True
          break
      if not ball_in_area_below_half_of_the_court:
        hit_net_event = 'LNF'
        return hit_net_event
        
  #check land to net from bottom player
  if (sign1 == False and sign2 == True) or (sign1 == True and sign2 == True): 
    ball_position = (trajectory[current_index][1],trajectory[current_index][2])
    event_in_net_area1 = is_inside_area(ball_position, right_half_double_line, left_half_double_line, right_net_end, left_net_end, image_width, image_height)
    ball_position = (ball_position[0], ball_position[1] + 10) ##tolerance due to imperfection of detection, move little bit down
    event_in_net_area2 = is_inside_area(ball_position, right_half_double_line, left_half_double_line, right_net_end, left_net_end, image_width, image_height)
    
    if event_in_net_area1 == 'in' or event_in_net_area2 == 'in': 
      
      is_ball_over_net = False
      for i in range(current_index, len(trajectory)):
        ball_position = (trajectory[i][1],trajectory[i][2])
        
        current_frame_index = trajectory[i][0]
        #net
        right_net_end = corner_points_per_images[current_frame_index][23]
        left_net_end = corner_points_per_images[current_frame_index][21]
        middle_net = corner_points_per_images[current_frame_index][22]
        #top base line
        left_corner_base_line_top_double = corner_points_per_images[current_frame_index][13]
        right_corner_base_line_top_double = corner_points_per_images[current_frame_index][10]
        half_base_line_top = corner_points_per_images[current_frame_index][18]
        #bottom base line
        left_corner_base_line_bottom_double = corner_points_per_images[current_frame_index][3]
        right_corner_base_line_bottom_double = corner_points_per_images[current_frame_index][0]
        half_base_line_bottom = corner_points_per_images[current_frame_index][17]

        ball_in_left_area1 = is_inside_area(ball_position, middle_net, left_net_end, half_base_line_top, left_corner_base_line_top_double, image_width, image_height)
        ball_in_right_area1 = is_inside_area(ball_position, right_net_end, middle_net, right_corner_base_line_top_double, half_base_line_top, image_width, image_height)
        ball_position = (ball_position[0], ball_position[1] + 10) ##tolerance due to imperfection of detection, move little bit down
        ball_in_left_area2 = is_inside_area(ball_position, middle_net, left_net_end, half_base_line_top, left_corner_base_line_top_double, image_width, image_height)
        ball_in_right_area2 = is_inside_area(ball_position, right_net_end, middle_net, right_corner_base_line_top_double, half_base_line_top, image_width, image_height)
        
        if (ball_in_left_area1 == 'in' and ball_in_left_area2 == 'in') or (ball_in_right_area1 == 'in' and ball_in_right_area2 == 'in'): #micek prekrocil pasku
          is_ball_over_net = True
          break
      if not is_ball_over_net:
        hit_net_event = 'LNN'
        return hit_net_event
  return hit_net_event

def compute_speed_of_serve(trajectory_index_of_serve_landing, trajectory_index_of_serve_hit, trajectory, players_detections, serve_event_name,\
                           homographies_image_to_artificial_court_per_frame, len_of_base_line_in_homography, margin_in_homography,fps):
  """
    Description:
      Compute speed of serve from serve hit and landing
    Parameters:
      trajectory_index_of_serve_landing                   int                                 : trajectory index of serve landing
      trajectory_index_of_serve_hit                       int                                 : trajectory index of serve hit
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      homographies_image_to_artificial_court_per_frame    list(homography)                    : list of homographies per image from cv2.findHomography()
      len_of_base_line_in_homography                      int                                 : length of baseline in homographu
      margin_in_homography                                int                                 : margins around court in homography
      fps                                                 int                                 : frames per second of video
    Returns:
      speed                                                 int                                 : mean serve speed between serve hit and serve land
  """                         
                           
                           
  land_frame_index = trajectory[trajectory_index_of_serve_landing][0]
  hit_frame_index = trajectory[trajectory_index_of_serve_hit][0]
  num_frames_between_events = land_frame_index - hit_frame_index
  real_time_between_events = num_frames_between_events/fps
  
  land_position = (np.float32(trajectory[trajectory_index_of_serve_landing][1]),np.float32(trajectory[trajectory_index_of_serve_landing][2]))
  if serve_event_name == 'SFL' or serve_event_name == 'SFR':
    player_position = np.float32(get_bottom_center_from_bounding_box(players_detections['player_top'][hit_frame_index]))
  elif serve_event_name == 'SNL' or serve_event_name == 'SNR':
    player_position = np.float32(get_bottom_center_from_bounding_box(players_detections['player_bottom'][hit_frame_index]))
  homography_for_land = homographies_image_to_artificial_court_per_frame[land_frame_index]
  homography_for_hit = homographies_image_to_artificial_court_per_frame[hit_frame_index]

  land_position_transformed = cv2.perspectiveTransform(np.array([[land_position]]), homography_for_land)
  player_position_transformed = cv2.perspectiveTransform(np.array([[player_position]]), homography_for_hit)

  #1097 real len of tennis base line in cm
  pix_to_cm_ratio = 1097/len_of_base_line_in_homography
  land_position_transformed = ((land_position_transformed[0][0][0] - margin_in_homography) * pix_to_cm_ratio, (land_position_transformed[0][0][1]-margin_in_homography) * pix_to_cm_ratio)
  player_position_transformed = ((player_position_transformed[0][0][0]-margin_in_homography) * pix_to_cm_ratio, (player_position_transformed[0][0][1]-margin_in_homography) * pix_to_cm_ratio)
  player_position_transformed = (player_position_transformed[0],player_position_transformed[1], 250)
  land_position_transformed = (land_position_transformed[0],land_position_transformed[1],0)
  dist = compute_distance(player_position_transformed,land_position_transformed)


  speed = dist / real_time_between_events
  speed = (speed /100000) * 3600 #/100 to meters / 1000 to km
  return speed

def find_game_ending(events_indexes_classificated, events_trajectory_indexes):
  """
    Description:
      Find tenis ending from events
    Parameters:
      events_indexes_classificated                        list(string)                        : list of events categories    
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
    Returns:
    game_ending_event                                     string                                  : name of game ending event
    index_of_game_ending_event                            int                                     : trajectory index of game end event
  """  
  game_ending_event = 'Undecided'
  index_of_game_ending_event = None
  
  SNI_events = ['SLFI']
  SNF_events = ['SLFO']
  SFI_events = ['SLNI']
  SFF_events = ['SLNO']
  HFR_events = ['HFR']
  HFL_events = ['HFL']
  HNR_events = ['HNR']
  HNL_events = ['HNL']
  
  #no hit net, would be detected in previous steps
  #check for double ball landing, must be before checking for outs, because could be land in - land out
  for i in range(len(events_indexes_classificated)-1):
    if events_indexes_classificated[i] == 'LFI' and events_indexes_classificated[i+1] == 'LFI':
      game_ending_event = 'winner_by_player_near'
      index_of_game_ending_event = events_trajectory_indexes[i+1]
      break
    elif events_indexes_classificated[i] == 'LFI' and events_indexes_classificated[i+1] == 'LFO':
      game_ending_event = 'winner_by_player_near'
      index_of_game_ending_event = events_trajectory_indexes[i+1]
      break
    elif events_indexes_classificated[i] == 'LNI' and events_indexes_classificated[i+1] == 'LNI':
      game_ending_event = 'winner_by_player_far'
      index_of_game_ending_event = events_trajectory_indexes[i+1]
      break
    elif events_indexes_classificated[i] == 'LNI' and events_indexes_classificated[i+1] == 'LNO':
      game_ending_event = 'winner_by_player_far'
      index_of_game_ending_event = events_trajectory_indexes[i+1]
      break
      
      
    elif events_indexes_classificated[i] == 'SLFI' and events_indexes_classificated[i+1] == 'LFO':
      game_ending_event = 'ace_by_player_near'
      index_of_game_ending_event = events_trajectory_indexes[i+1]
      break
    elif events_indexes_classificated[i] == 'SLNI' and events_indexes_classificated[i+1] == 'LNO':
      game_ending_event = 'ace_by_player_far'
      index_of_game_ending_event = events_trajectory_indexes[i+1]
      break
    elif events_indexes_classificated[i] == 'SLFI' and events_indexes_classificated[i+1] == 'LFI':
      game_ending_event = 'ace_by_player_near'
      index_of_game_ending_event = events_trajectory_indexes[i+1]
      break
    elif events_indexes_classificated[i] == 'SLNI' and events_indexes_classificated[i+1] == 'LNI':
      game_ending_event = 'ace_by_player_far'
      index_of_game_ending_event = events_trajectory_indexes[i+1]
      break

  #check for outs, no double landing detected
  if game_ending_event == 'Undecided':
    for i in range(len(events_indexes_classificated)):
      current_event = events_indexes_classificated[i]
      if current_event == 'LFO':
        game_ending_event = 'out_by_near_player'
        index_of_game_ending_event = events_trajectory_indexes[i]
        break
      elif current_event == 'LNO':
        game_ending_event= 'out_by_far_player'
        index_of_game_ending_event = events_trajectory_indexes[i]
        break
      elif current_event == 'SLFO':
        game_ending_event= 'serve_to_left_out_by_player_near'
        index_of_game_ending_event = events_trajectory_indexes[i]
        break
      elif current_event == 'SLNO':
        game_ending_event= 'serve_to_left_out_by_player_far'
        index_of_game_ending_event = events_trajectory_indexes[i]
        break
 
  #no out or double landing, check for winner 
  #check for winner normal or by serve
  if game_ending_event == 'Undecided':
    if events_indexes_classificated[-1] == 'LFI':
      game_ending_event = 'winner_by_player_near'
      index_of_game_ending_event = events_trajectory_indexes[-1]
    elif events_indexes_classificated[-1] == 'LNI':
      game_ending_event = 'winner_by_player_far'
      index_of_game_ending_event = events_trajectory_indexes[-1]
    elif events_indexes_classificated[-1] == 'SLNI':
      game_ending_event = 'ace_by_player_far'
      index_of_game_ending_event = events_trajectory_indexes[-1]
    elif events_indexes_classificated[-1] == 'SLFI':
      game_ending_event = 'ace_by_player_near'
      index_of_game_ending_event = events_trajectory_indexes[-1]
      

  #end by hit
  if game_ending_event == 'Undecided':
    if events_indexes_classificated[-1] in HFR_events or events_indexes_classificated[-1] in HFL_events:
      game_ending_event = 'out_by_far_player'
      index_of_game_ending_event = events_trajectory_indexes[-1]
    elif events_indexes_classificated[-1] in HNR_events or events_indexes_classificated[-1] in HNL_events:
      game_ending_event = 'out_by_near_player'
      index_of_game_ending_event = events_trajectory_indexes[-1]
  
  return game_ending_event, index_of_game_ending_event
  
def clean_trajectory_where_court_not_detected(trajectory, corner_points_per_images, players_detections):
  """
    Description:
      Remove detections where ball is not detected
    Parameters:
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
    Returns:
    trajectory                                            list(tuple[int,int,int])            : cleaned trajectory
  """
  
  for i in range(len(trajectory)-1,-1,-1):
    players_detected = True
    if players_detections['player_top'][trajectory[i][0]].xmin == 0 and players_detections['player_top'][trajectory[i][0]].xmax == 0 and players_detections['player_top'][trajectory[i][0]].ymin == 0 and players_detections['player_top'][trajectory[i][0]].ymax == 0:
      players_detected = False
    if corner_points_per_images[trajectory[i][0]] is None or players_detected == False:
      trajectory.pop(i)
  return trajectory
  
def is_inside_baselines_extended_area(ball_position,corner_points_per_images,current_frame_index,image_width, image_height):
  """
    Description:
      Extends baselines to end of image and check if ball is inside this area
    Parameters:
      ball_position                                       int,int                             : x,y coordinates of ball detection to check
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      current_frame_index                                 int                                 : frame index of detection
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
                                                          bool                                : True if ball is inside baseline extended area
  """
  left_top = corner_points_per_images[current_frame_index][13]
  right_top = corner_points_per_images[current_frame_index][10]

  left_bottom = corner_points_per_images[current_frame_index][3]
  right_bottom = corner_points_per_images[current_frame_index][0]
  
  vector_left1 = (left_top[0] - right_top[0], left_top[1] - right_top[1])
  vector_right1 = (right_top[0] - left_top[0], right_top[1] - left_top[1])

  vector_left2 = (left_bottom[0] - right_bottom[0], left_bottom[1] - right_bottom[1])
  vector_right2 = (right_bottom[0] - left_bottom[0], right_bottom[1] - left_bottom[1])

  left_end1 = get_end_point_from_vector_and_point(vector_left1, left_top, image_width, image_height)
  right_end1 = get_end_point_from_vector_and_point(vector_right1, right_top, image_width, image_height)

  left_end2 = get_end_point_from_vector_and_point(vector_left2, left_bottom, image_width, image_height)
  right_end2 = get_end_point_from_vector_and_point(vector_right2, right_bottom, image_width, image_height)

  inside_baseline_extended_area = is_inside_area(ball_position, right_end2, left_end2, right_end1, left_end1, image_width, image_height)

  return inside_baseline_extended_area  
  
  
  
def last_filtration(events_indexes_classificated, events_trajectory_indexes, trajectory, players_detections, corner_points_per_images,image_width, image_height,tenis_game_ending ):
  """
    Description:
      Delete hits if ball is not inside court area after that hit
      Filter sequence for more than one ball land between player hits
      Delete hits if they are not changing between two players 
      After last hit keep max two ball lands
      Before LNF only land far or hit far, Before LNN only land ner or hit near, if not remove until it is
    Parameters:
      events_indexes_classificated                        list(string)                        : list of events categories
      events_trajectory_indexes                           list(int)                           : list of events indexes to trajectory
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
    events_indexes_classificated                          list(string)                        : if event is removed, then is removed from list, if it is changed, then it is changed in the list    
    events_trajectory_indexes                             list(int)                           : if event is removed, index is removed from list
  """

  while True:
    #Delete hits if ball is not inside court area after that hit
    size_before = len(events_indexes_classificated)
    for i in range(len(trajectory)-1,-1,-1):
      ball_position = (trajectory[i][1],trajectory[i][2])
      current_frame_index = trajectory[i][0]
      res = is_inside_baselines_extended_area(ball_position,corner_points_per_images,current_frame_index,image_width, image_height)
      if res == 'in':
        first_index_in = i
        break
    for i in range(len(events_trajectory_indexes)-1,-1,-1):
      if events_trajectory_indexes[i] > first_index_in and (events_indexes_classificated[i] == 'hit_far' or events_indexes_classificated[i] == 'hit_near'):
        events_trajectory_indexes.pop(i)
        events_indexes_classificated.pop(i)
    
    
    
    #Delete hits if they are not changing between two players 
    to_remove = []
    next_hit_near = True
    next_hit_far = True
    index_of_previous_hit_near = 0
    index_of_previous_hit_far = 0
    for i in range(len(events_trajectory_indexes)-1,-1,-1):
      if events_indexes_classificated[i] == 'hit_near':
        if next_hit_near == False:
          dist1 = get_dist_to_player(trajectory, events_trajectory_indexes[i],players_detections, 'player_near')
          dist2 = get_dist_to_player(trajectory, events_trajectory_indexes[index_of_previous_hit_near],players_detections, 'player_near')
          if dist1 > dist2:
            to_remove.append(i)
          else:
            to_remove.append(index_of_previous_hit_near)
        next_hit_near = False
        next_hit_far = True
        index_of_previous_hit_near = i  
        
        
      if events_indexes_classificated[i] == 'hit_far':
        if next_hit_far == False:
          dist1 = get_dist_to_player(trajectory, events_trajectory_indexes[i],players_detections, 'player_far')
          dist2 = get_dist_to_player(trajectory, events_trajectory_indexes[index_of_previous_hit_far],players_detections, 'player_far')
          if dist1 > dist2:
            to_remove.append(i)
          else:
            to_remove.append(index_of_previous_hit_far)
        next_hit_near = True
        next_hit_far = False
        index_of_previous_hit_far = i  
       
    to_remove = sorted(list(dict.fromkeys(to_remove)), reverse = False)
    
    for i in range(len(to_remove)-1,-1,-1):
      events_trajectory_indexes.pop(to_remove[i])
      events_indexes_classificated.pop(to_remove[i]) 
    
    
    event_index_of_last_hit_near = None
    event_index_of_last_hit_far = None
    for i in range(len(events_trajectory_indexes)-1,-1,-1):
      if events_indexes_classificated[i] == 'hit_far' or events_indexes_classificated[i] == 'SFR' or events_indexes_classificated[i] == 'SFL':
        event_index_of_last_hit_far = i
        break
   
    for i in range(len(events_trajectory_indexes)-1,-1,-1):
      if events_indexes_classificated[i] == 'hit_near' or events_indexes_classificated[i] == 'SNR' or events_indexes_classificated[i] == 'SNL':
        event_index_of_last_hit_near = i
        break  
    

    event_index_last_hit = None
    if event_index_of_last_hit_near is not None and event_index_of_last_hit_far is not None: 
      event_index_last_hit = max(event_index_of_last_hit_near,event_index_of_last_hit_far)
    elif event_index_of_last_hit_near is not None: 
      event_index_last_hit = event_index_of_last_hit_near
    elif event_index_of_last_hit_far is not None:   
      event_index_last_hit = event_index_of_last_hit_far
      
    #Filter sequence for more than one ball land between player hits  
    if event_index_last_hit is not None:
      for i in range(event_index_last_hit-1, 0,-1):
        if events_indexes_classificated[i] == 'land_near' and (events_indexes_classificated[i-1] == 'land_near' or events_indexes_classificated[i-1] == 'serve_land_near_left' or events_indexes_classificated[i-1] == 'serve_land_near_right'):
          events_trajectory_indexes.pop(i)
          events_indexes_classificated.pop(i)
          
      for i in range(event_index_last_hit-1, 0,-1):
        if events_indexes_classificated[i] == 'land_far' and (events_indexes_classificated[i-1] == 'land_far' or events_indexes_classificated[i-1] == 'serve_land_far_left' or events_indexes_classificated[i-1] == 'serve_land_far_right'):
          events_trajectory_indexes.pop(i)
          events_indexes_classificated.pop(i)
          
      #After last hit keep max two ball lands    
      cnt_lands = 0
      to_remove = []
      for i in range(event_index_last_hit, len(events_indexes_classificated)):
        if events_indexes_classificated[i] == 'land_far' or events_indexes_classificated[i] == 'land_near' or events_indexes_classificated[i] == 'serve_land_near_left' or events_indexes_classificated[i] == 'serve_land_near_right' or events_indexes_classificated[i] == 'serve_land_far_left' or events_indexes_classificated[i] == 'serve_land_far_right':
          cnt_lands = cnt_lands + 1
          if cnt_lands > 2:
            to_remove.append(i)
            
      for i in range(len(to_remove)-1,-1,-1):
          events_trajectory_indexes.pop(to_remove[i])
          events_indexes_classificated.pop(to_remove[i])    
    
    #Before LNF only land far or hit far, Before LNN only land ner or hit near, if not remove until it is
    if tenis_game_ending != 'Undecided':
      if tenis_game_ending[1] == 'LNN': #dat tam hit net do eventu
        for i in range(len(events_indexes_classificated)-2,-1,-1):
          if events_indexes_classificated[i] == 'hit_near' or events_indexes_classificated[i] == 'land_near':
            break
          events_trajectory_indexes.pop(i)
          events_indexes_classificated.pop(i)
 
          
      elif tenis_game_ending[1] == 'LNF':
        for i in range(len(events_indexes_classificated)-2,-1,-1):
          if events_indexes_classificated[i] == 'hit_far' or events_indexes_classificated[i] == 'land_far':
            break
          events_trajectory_indexes.pop(i)
          events_indexes_classificated.pop(i)
    
    if size_before == len(events_indexes_classificated):
      break
      
  return events_indexes_classificated, events_trajectory_indexes   
def check_lnf_from_serve_toss(tenis_game_ending, events_indexes_classificated, trajectory, corner_points_per_images, players_detections, image_width, image_height):   
  """
    Description:
      Filter sequence if LNF is not from serve toss of near player
    Parameters:
      tenis_game_ending                                   tuple(int,string) or string         : index and type of game ending, checks if it is land to net, if not found then value is 'Undecided'
      events_indexes_classificated                        list(string)                        : list of events categories
      trajectory                                          list(tuple[int,int,int])            : list of detections, indexing [detection id][i], i=0 frame index, i = 1 x, i =2 y
      corner_points_per_images                            list(list(tuple[int,int]))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      players_detections                                  list(list(bounding box))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      image_width                                         int                                 : image widht
      image_height                                        int                                 : image height
    Returns:
    events_indexes_classificated                          list(string)                        : if event is removed, then is removed from list, if it is changed, then it is changed in the list    
    events_trajectory_indexes                             list(int)                           : if event is removed, index is removed from list
  """ 
  is_serve_to_net = False
  from_serve_toss_of_near_player = False
  is_false_LNF = False
  if tenis_game_ending[1] == 'LNF':
    is_serve_to_net = True
    for i in range(len(events_indexes_classificated)-1):
      if events_indexes_classificated[i] == 'land_near' and events_indexes_classificated[i+1] == 'hit_near':
        is_serve_to_net = False
  else:
    return False, None
    
  if is_serve_to_net == True:#should be far serve to net, check
    current_index = tenis_game_ending[0]
    for i in range(current_index, -1,-1):
      ball_position = (trajectory[i][1],trajectory[i][2])
      frame_index = trajectory[i][0]
      left_corner_base_line_bottom_double = corner_points_per_images[frame_index][3]
      right_corner_base_line_bottom_double = corner_points_per_images[frame_index][0]
      #half line
      right_half_double_line = corner_points_per_images[frame_index][19]
      left_half_double_line = corner_points_per_images[frame_index][20]
      ball_in_area_below_half1 = is_inside_area(ball_position, right_corner_base_line_bottom_double, left_corner_base_line_bottom_double, right_half_double_line, left_half_double_line, image_width, image_height)
      if ball_in_area_below_half1 == 'in': #micek v dolni polovine, nemuze byt podani far
        from_serve_toss_of_near_player = True
        break
    if from_serve_toss_of_near_player == False:
      ball_position = (trajectory[0][1],trajectory[0][2])
      frame_index = trajectory[0][0]
      ball_y = ball_position[1]
      player_far_y = players_detections['player_top'][frame_index].ymin
      net_middle_point_y = corner_points_per_images[frame_index][22][1]
      dist_ball_player = abs(player_far_y - ball_y)
      dist_ball_net = abs(net_middle_point_y - ball_y)
      if dist_ball_player > dist_ball_net:
         from_serve_toss_of_near_player = True
    if from_serve_toss_of_near_player:
      is_false_LNF = True
      
  return is_false_LNF, tenis_game_ending[0] 
  
  
def get_events_sequence(trajectory_orig, trajectory_filtered, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, image_width, image_height, len_of_base_line_in_homography, margin_in_homography, fps):
  """
    Description:
      Find sequence of events in tennis game
    Parameters:
      trajectory_orig                                     (list(int,int,int))                   : trajectory represented as frame,x,y 
      trajectory_filtered                                 (list(int,int,int))                   : trajectory filtrated for example by mean filter, represented as frame,x,y 
      players_detections                                  (list(list(bounding box)))            : list of players bounding boxes per image, indexing [player][image]. player = 'player_top'/'player_bottom', bounding box values xmax,xmin,ymax,ymin
      corner_points_per_images                            (list(list(tuple[int,int])))          : list of court points per image, indexing [image][point][i]  i = 0 x, i = 1 y
      homographies_image_to_artificial_court_per_frame    (list(homography))                    : list of homographies per image from cv2.findHomography()
      image_width                                         (int)                                 : image widht
      image_height                                        (int)                                 : image height
      len_of_base_line_in_homography                      (int)                                 : base horizontal line length in pixels for articifical court model
      margin_in_homography                                (int)                                 : margin in pixels around tennis court model for articifical court model
      fps                                                 (fps)                                 : fps of source video
    Returns:
    events_indexes_classificated_with_frame_num           (list(int,string))                    : list of annotated events, format frame,event
    tenis_game_ending                                     (int,string)                          : frame of event ending correct tennis game and name of that event
    speed_of_serve                                        (int)                                 : speed of serve in km/h
  """
  
  
  width = image_width
  height = image_height
  
  #clean trajectories if two neighboring detections has same coords
  for i in range(len(trajectory_orig)-1,0,-1):
    if (trajectory_orig[i][1] == trajectory_orig[i-1][1]) and (trajectory_orig[i][2] == trajectory_orig[i-1][2]):
      trajectory_orig.pop(i)
      
  for i in range(len(trajectory_filtered)-1,0,-1):
    if (trajectory_filtered[i][1] == trajectory_filtered[i-1][1]) and (trajectory_filtered[i][2] == trajectory_filtered[i-1][2]):
      trajectory_filtered.pop(i)
      
  ###########################################################RUN1############################################################################################################################

  
  trajectory = copy.deepcopy(trajectory_filtered)
  points_artificial_court = get_corner_points_for_tennis_court(len_of_base_line_in_homography,margin_in_homography)
  excluded_for_hit_net = []
  trajectory = clean_trajectory_where_court_not_detected(trajectory, corner_points_per_images,players_detections)
  if trajectory == []:
    return None, None, None
  repeat = True
  while repeat:
    repeat = False
    events_trajectory_indexes = []
    #detect event candidates by angles
    events_trajectory_indexes_with_angles = detect_event_candidates(trajectory)
    events_trajectory_indexes = filter_events_belonging_to_one_direction_shift(events_trajectory_indexes_with_angles)
    
    tenis_game_ending = 'Undecided'

    #find first land to net
    for i in range(len(events_trajectory_indexes)):
      if events_trajectory_indexes[i] in excluded_for_hit_net:
        continue
      hit_net_event = find_land_to_net(i,trajectory,events_trajectory_indexes, corner_points_per_images, players_detections, image_width, image_height)
      
      if hit_net_event == 'LNN' or hit_net_event == 'LNF': #remove all events after hit net event including itself, set ending of this tenis game as hit net
        tenis_game_ending = (events_trajectory_indexes[i],hit_net_event)
        for j in range(len(events_trajectory_indexes)-1,i-1,-1):#remove events after hit net
          events_trajectory_indexes.pop(j)
        break

    if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF': #put land net event index back for filtering as sentinel of other events, but alnd net is not filtered
        events_trajectory_indexes.append(tenis_game_ending[0])
        number_of_events_to_classify = len(events_trajectory_indexes) - 1     
    else:
      number_of_events_to_classify = len(events_trajectory_indexes)
    
    #classify events to categories land_far, hit_far, land_near, hit_near
    events_indexes_classificated = []
    for i in range(number_of_events_to_classify):
      event_type = classify_event(i,events_trajectory_indexes, trajectory, players_detections,corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court)
      events_indexes_classificated.append(event_type)
    
    
    if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF': 
      events_indexes_classificated.append(tenis_game_ending[1])


        
    while True:
      if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF': # do not filter land to net
        indexes_to_filter = [i for i in range(len(events_indexes_classificated)-2,-1,-1)]
      else:
        indexes_to_filter = [i for i in range(len(events_indexes_classificated)-1,-1,-1)]
      #filtr events by tests
      size_before = len(events_indexes_classificated)
      events_trajectory_indexes,events_indexes_classificated = filter_events(indexes_to_filter,events_trajectory_indexes, events_indexes_classificated, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)
      if size_before == len(events_indexes_classificated) or (len(events_indexes_classificated) == 1 and (tenis_game_ending == 'LNN' or tenis_game_ending == 'LNF')):
        break       
        
    #if land to net, pop it, not part of automata filtering    
    if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF': 
      events_trajectory_indexes.pop(-1)
      events_indexes_classificated.pop(-1)
      
      
    #automata filtering  
    automata = get_automata()
    events_trajectory_indexes, events_indexes_classificated = filter_events_by_automata(automata, events_trajectory_indexes, events_indexes_classificated, trajectory,players_detections,corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)
    
    #put land to net back
    if tenis_game_ending != 'Undecided':
      if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF': 
        events_trajectory_indexes.append(tenis_game_ending[0])
        events_indexes_classificated.append(tenis_game_ending[1])  
        
    #check for false LNF, if detected run again with LNF on that index excluded
    repeat, to_exclude = check_lnf_from_serve_toss(tenis_game_ending, events_indexes_classificated, trajectory, corner_points_per_images, players_detections, image_width, image_height)  
    if repeat == True:
      excluded_for_hit_net.append(to_exclude)   
   
  #find serve landing
  events_indexes_classificated, events_trajectory_indexes, serve_land_index, serve_event, tenis_game_ending,serve_land_event_index = find_land_serve_event(events_indexes_classificated,events_trajectory_indexes,trajectory,players_detections,corner_points_per_images, tenis_game_ending, image_width, image_height)
  land_serve_event = None
  if serve_land_event_index is not None:
    land_serve_event = events_indexes_classificated[serve_land_event_index]


    
    
  ###########################################################RUN2############################################################################################################################
  
  ########################################################### same as run 1 but with different trajectory####################################################################################
  
  trajectory = copy.deepcopy(trajectory_orig)
  points_artificial_court = get_corner_points_for_tennis_court(len_of_base_line_in_homography,margin_in_homography)
  excluded_for_hit_net = []

  trajectory = clean_trajectory_where_court_not_detected(trajectory, corner_points_per_images,players_detections)
  if trajectory == []:
    return None, None, None
  repeat = True
  while repeat:
    repeat = False
    events_trajectory_indexes = []
    #detect event candidates by angles
    events_trajectory_indexes_with_angles = detect_event_candidates(trajectory)
    events_trajectory_indexes = filter_events_belonging_to_one_direction_shift(events_trajectory_indexes_with_angles)
    
    tenis_game_ending = 'Undecided'    
    all_hit_net_candidates = []
    #find all land to net candidates
    for i in range(len(events_trajectory_indexes)):
      if events_trajectory_indexes[i] in excluded_for_hit_net:
        continue
      hit_net_event = find_land_to_net(i,trajectory,events_trajectory_indexes, corner_points_per_images, players_detections, image_width, image_height)
      if hit_net_event == 'LNN' or hit_net_event == 'LNF': #remove all events after hit net event including itself, set ending of this tenis game as hit ne
        all_hit_net_candidates.append([events_trajectory_indexes[i],hit_net_event])
    
    if len(all_hit_net_candidates) > 0: #get first net hit
      tenis_game_ending = (all_hit_net_candidates[0][0],all_hit_net_candidates[0][1])
      for j in range(len(events_trajectory_indexes)-1,-1,-1):#remove events after hit net
        if events_trajectory_indexes[j] > tenis_game_ending[0]:
          events_trajectory_indexes.pop(j)
        
    
    if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF': #put land net event index back for filtering as sentinel of other events, but alnd net is not filtered
        events_trajectory_indexes.append(tenis_game_ending[0])
        number_of_events_to_classify = len(events_trajectory_indexes) - 1
        
    else:
      number_of_events_to_classify = len(events_trajectory_indexes)
    
    #classify events to categories land_far, hit_far, land_near, hit_near
    events_indexes_classificated = []
    for i in range(number_of_events_to_classify):
      event_type = classify_event(i,events_trajectory_indexes, trajectory, players_detections,corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court)
      events_indexes_classificated.append(event_type)
    
    if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF': 
      events_indexes_classificated.append(tenis_game_ending[1])
      
        
    while True:
      if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF': # do not filter land to net
        indexes_to_filter = [i for i in range(len(events_indexes_classificated)-2,-1,-1)]
      else:
        indexes_to_filter = [i for i in range(len(events_indexes_classificated)-1,-1,-1)]
      
      #filter events by tests
      size_before = len(events_indexes_classificated)
      events_trajectory_indexes,events_indexes_classificated = filter_events(indexes_to_filter,events_trajectory_indexes, events_indexes_classificated, trajectory, players_detections, corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)
      if size_before == len(events_indexes_classificated) or (len(events_indexes_classificated) == 1 and (tenis_game_ending == 'LNN' or tenis_game_ending == 'LNF')):
        break       
    #if land to net, pop it, not part of automata filtering       
    if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF': 
      events_trajectory_indexes.pop(-1)
      events_indexes_classificated.pop(-1)

    #automata filtering      
    automata = get_automata()
    events_trajectory_indexes, events_indexes_classificated = filter_events_by_automata(automata, events_trajectory_indexes, events_indexes_classificated, trajectory,players_detections,corner_points_per_images, homographies_image_to_artificial_court_per_frame, points_artificial_court, image_width, image_height)
   
    #put land to net back
    if tenis_game_ending[1] == 'LNN' or tenis_game_ending[1] == 'LNF': 
      events_trajectory_indexes.append(tenis_game_ending[0])
      events_indexes_classificated.append(tenis_game_ending[1])    
      
    #check for false LNF, if detected run again with LNF on that index excluded    
    repeat, to_exclude = check_lnf_from_serve_toss(tenis_game_ending, events_indexes_classificated, trajectory, corner_points_per_images, players_detections, image_width, image_height)  
    if repeat == True:
      excluded_for_hit_net.append(to_exclude)  
        
  ########################################################### different from  run 1 ############################################################################################################################ 
  
  #serve land index from previous run 
  serve_land_index_previous = serve_land_index
  
  serve_found_in_run2 = False
  #if serve event detected in run1
  if land_serve_event is not None:
    #if in run1 detected serve to net
    if land_serve_event == 'LNN' or land_serve_event == 'LNF':
      min_index = None
      min_distance = float('inf')
      land_index_in_previous_run = serve_land_index
      
      if serve_event == 'SFR' or serve_event == 'SFL':
        for i in range(len(all_hit_net_candidates)):
          if (all_hit_net_candidates[i][1] == 'LNF') and (min_distance > abs(all_hit_net_candidates[i][0] - land_index_in_previous_run)):
            min_distance = abs(all_hit_net_candidates[i][0] - land_index_in_previous_run)
            min_index = all_hit_net_candidates[i][0]
      if serve_event == 'SNR' or serve_event == 'SNL':
        for i in range(len(all_hit_net_candidates)):
          if (all_hit_net_candidates[i][1] == 'LNN') and (min_distance > abs(all_hit_net_candidates[i][0] - land_index_in_previous_run)):
            min_distance = abs(all_hit_net_candidates[i][0] - land_index_in_previous_run)
            min_index = all_hit_net_candidates[i][0]
      
      
      serve_land_index = min_index  
      if serve_land_index is None:
        events_trajectory_indexes = [serve_land_index_previous]
        tenis_game_ending = (serve_land_index_previous,land_serve_event)
      else:
        events_trajectory_indexes = [serve_land_index]
        tenis_game_ending = (serve_land_index,land_serve_event)
        
        
      events_indexes_classificated = [land_serve_event]
    #if in run1 detected serve   
    else:
      min_index = None
      min_distance = float('inf')
      land_index_in_previous_run = serve_land_index
      #in run1 detected serve  from far player, try to find closest land_near in current run2 to land_near found in run1
      if serve_event == 'SFR' or serve_event == 'SFL':
        for i in range(len(events_indexes_classificated)):
          if (events_indexes_classificated[i] == 'land_near') and (min_distance > abs(events_trajectory_indexes[i] - land_index_in_previous_run)):
            min_distance = abs(events_trajectory_indexes[i] - land_index_in_previous_run)
            min_index = events_trajectory_indexes[i]
      #in run1 detected serve  from near player, try to find closest land_far in current run2 to land_far found in run1    
      elif serve_event == 'SNR' or serve_event == 'SNL':
        for i in range(len(events_indexes_classificated)):
          if (events_indexes_classificated[i] == 'land_far') and (min_distance > abs(events_trajectory_indexes[i] - land_index_in_previous_run)):
            min_distance = abs(events_trajectory_indexes[i] - land_index_in_previous_run)
            min_index = events_trajectory_indexes[i]

      serve_land_index = min_index
  #serve land not found in run1, try to find him in run2
  else:
    events_indexes_classificated, events_trajectory_indexes, serve_land_index, serve_event, tenis_game_ending,serve_land_event_index = find_land_serve_event(events_indexes_classificated,events_trajectory_indexes,trajectory,players_detections,corner_points_per_images, tenis_game_ending, image_width, image_height)    
    #if serve land found in run2
    if serve_land_event_index is not None:
      land_serve_event = events_indexes_classificated[serve_land_event_index]
      serve_found_in_run2 = True
    else:
      land_serve_event = None
  #if serve land found in run1 or run2
  if land_serve_event is not None:
    #in this run2 not found same serve land as in run1, take trajectory index of serve land from run1
    if serve_land_index is None:
      serve_land_index = serve_land_index_previous
      
    #clean events before serve land, so not false lands will be examined in serve land finding in current trajectory
    for i in range(len(events_indexes_classificated)-1,-1,-1):
      if events_trajectory_indexes[i] < serve_land_index:
        events_trajectory_indexes.pop(i)
        events_indexes_classificated.pop(i)
     
    #check if serve to net still correct after filtering trajectory events
    if land_serve_event  == 'LNN' or land_serve_event  == 'LNF':
      land_serve_event = find_land_to_net(len(events_trajectory_indexes) -1,trajectory,events_trajectory_indexes, corner_points_per_images, players_detections, image_width, image_height)
      if land_serve_event == False: 
        events_trajectory_indexes.pop(i)
        events_indexes_classificated.pop(i)
        serve_found_in_run2 = False
        land_serve_event = None
        tenis_game_ending = 'Undecided'
    #if serve land index was not examined in run2, do it now after cleaning
    if serve_found_in_run2 == False:
      events_indexes_classificated, events_trajectory_indexes, serve_land_index, serve_event, tenis_game_ending,serve_land_event_index = find_land_serve_event(events_indexes_classificated,events_trajectory_indexes,trajectory,players_detections,corner_points_per_images, tenis_game_ending, image_width, image_height)    
  
  #serve land found, then find serve hit
  if serve_land_index is not None:  
    serve_trajectory_index = find_serve_event_from_landing_spot(serve_land_index, serve_event, events_indexes_classificated,events_trajectory_indexes,trajectory,players_detections,corner_points_per_images, tenis_game_ending, image_width, image_height)
    if serve_trajectory_index is not None:
      events_indexes_classificated,events_trajectory_indexes = include_serve_to_events(serve_trajectory_index,serve_event, events_indexes_classificated,events_trajectory_indexes)

      
  #additional filtration    
  events_indexes_classificated,events_trajectory_indexes = last_filtration(events_indexes_classificated, events_trajectory_indexes, trajectory, players_detections, corner_points_per_images,image_width, image_height,tenis_game_ending)
  
  #make more specific annotations
  events_indexes_classificated = make_higher_anotations(events_indexes_classificated,events_trajectory_indexes, trajectory, players_detections, corner_points_per_images, image_width, image_height, homographies_image_to_artificial_court_per_frame,points_artificial_court, True)
  
  #find serve speed
  speed_of_serve = 'x'
  if serve_land_index is not None:
    if events_indexes_classificated[1] != 'LNF' and events_indexes_classificated[1] != 'LNN':
      speed_of_serve = compute_speed_of_serve(events_trajectory_indexes[1], events_trajectory_indexes[0], trajectory, players_detections, events_indexes_classificated[0], homographies_image_to_artificial_court_per_frame,len_of_base_line_in_homography,margin_in_homography, fps)
      speed_of_serve = int (speed_of_serve)
   
  #find type of game ending 
  if tenis_game_ending != 'Undecided' and tenis_game_ending is not None:
    index_of_game_ending_event = tenis_game_ending[0]
    game_ending_event = tenis_game_ending[1]
    pass
  else:
    game_ending_event, index_of_game_ending_event = find_game_ending(events_indexes_classificated, events_trajectory_indexes)

  ending_event_trajectory_index = index_of_game_ending_event
  
  #transform to format (frame index,event name)
  events_indexes_classificated_with_frame_num = []
  for i in range(len(events_indexes_classificated)):
    events_indexes_classificated_with_frame_num.append((trajectory[events_trajectory_indexes[i]][0], events_indexes_classificated[i]))
    
  return events_indexes_classificated_with_frame_num, tenis_game_ending,  speed_of_serve
