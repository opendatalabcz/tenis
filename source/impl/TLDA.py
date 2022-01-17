from joblib import dump, load
import math
import copy
import pandas as pd
import os
import cv2
import numpy as np
import copy
import math
import random
from joblib import dump, load
import functools
import sys

sys.path.append('.')
from utils import Path, rotate_vector, get_angle2, get_intersection_of_two_lines_segments, compute_distance_abs, is_inside_area, get_end_point_from_vector_and_point,is_inside_middle_lines_extended_area

VERSION = 'C'#'PyCUDA', 'Python', 'C_and_PyCUDA
PATH_TO_FW_SHARED_LIBRARY = './fw.dll'
if VERSION == 'C':
  import ctypes
if VERSION == 'PyCUDA':
  import pycuda.driver as cuda
  from pycuda.compiler import SourceModule
if VERSION == 'C_and_PyCUDA':
  import pycuda.driver as cuda
  from pycuda.compiler import SourceModule
  import ctypes
from time import process_time, time, sleep


INF = np.int32(100000000)#sys.float_info.max

#algorithm parameters
#image shape
widht = 1280
height =  720
DELTA_T = 1
#max and min distances between detections to connect them inside window
MAX_DIST = 130
MIN_DIST = 2

windows_size = 15
window_shift = 3

#how many continous frames can have missed detection in path
limit_VCs = 2
#limit minimal value of score that can be added for new detection
MIN_SCORE = -5
#computational time parameters
paths_to_keep_to_next_round = 10
upper_limit_on_paths_per_window = 6 #6
#pruning parameters
x_vs_y_movement_threshold = 0.3
max_big_shifts_to_length_ratio = 0.1


LINE_SIZE = 5

#source of floyd warshall on gpu 
#https://saadmahmud14.medium.com/parallel-programming-with-cuda-tutorial-part-4-the-floyd-warshall-algorithm-5e1281c46bf6


class Path:
  """Class for representing trajectories in video

    Class for representing trajectories in video. Trajectories are represented as coordinates x,y of detections with information about frame where detections were located.

    Attributes:
      score                             (float)                     : score of path, smaller is better. Can be weighted by smoothness for example.
      path                              (list(tuple[int,int]))      : detections in path in format (index of frame in window, index of detection in frame)
      path_coords                       (list(tuple[int,int,int]))  : detections in path in format (index of frame global, x coordinate, y coordinate)
      predictions                       (list(tuple[int,int]))      : for DEBUG, predictions made for evaluating score of new detection
      virtual_nodes_counter             (int)                       : counting how many continous frames have missing detection, if detection is found counter is set to 0
      score_arr                         (list(float))               : for DEBUG, array of scores for detections added to path 
      score_without_shifts_weightings   (float)                     : score without any weighting only by motion model defined in paper
      cnt_dir_shifts                    (int)                       : number of big changes of direction in each three continous detections in path
      x_movement                        (float)                     : accumulated movement in x direction
      y_movement                        (float)                     : accumulated movement in y direction
  """
  def __init__(self,start,start2,start3 = []):
    self.score =  float("inf")
    self.path = [start]
    self.path_coords = [start2] 
    self.predictions = start3 
    self.virtual_nodes_counter = 0
    self.score_arr = [0,0,0] #first three detections has score 0
    self.score_without_shifts_weightings =  float("inf")
    self.cnt_dir_shifts = 0 
    self.x_movement = 0
    self.y_movement = 0

def compute_score(path, next_detection, multiplicator = 1): 
  """
    Description:
      Compute score based on prediction from last 3 detections in trajectory and detection to be added to trajectory. 
    Parameters:
      path                     (list(tuple[int,int,int])): list of detections in path, format (frame,x,y)
      next_detection           (tuple[int,int])          : coordinates x-y of detection to be added to trajectory
      multiplicator            (int)                     : number of frames the next_detection is ahead of detection3. 
    Returns:
      actual_score             (float)                   : Score based on distance and angle between next_detection and prediction from detection1-3
    """
  detection1 = (path.path_coords[-3][1],path.path_coords[-3][2])
  detection2 = (path.path_coords[-2][1],path.path_coords[-2][2])
  detection3 = (path.path_coords[-1][1],path.path_coords[-1][2])
  return compute_score2(detection1, detection2, detection3, next_detection, multiplicator)
 
  
def compute_score2(detection1, detection2, detection3, next_detection, multiplicator = 1): 
  """
    Description:
      Compute score based on prediction from last 3 detections in trajectory and detection to be added to trajectory. 
    Parameters:
      detection1               (tuple[int,int])          : first detection
      detection2               (tuple[int,int])          : second detection
      detection3               (tuple[int,int])          : third detection
      next_detection           (tuple[int,int])          : coordinates x-y of detection to be added to trajectory
      multiplicator            (int)                     : number of frames the next_detection is ahead of detection3. 
    Returns:
      actual_score             (float)                   : Score based on distance and angle between next_detection and prediction from detection1-3
    """
  prediction_x, prediction_y = get_prediction(detection1,detection2,detection3,multiplicator)
  dist = compute_distance_abs((prediction_x,prediction_y),next_detection)

  if dist > MIN_DIST:
    actual_score = max( math.log( (dist - MIN_DIST ) / (MAX_DIST - MIN_DIST)), MIN_SCORE )  - math.log( (MAX_DIST/2 - MIN_DIST ) / (MAX_DIST - MIN_DIST)) #pulka max dist se lame
  else:
    actual_score = MIN_SCORE

  vector_1 = [prediction_x - detection3[0], prediction_y - detection3[1]]
  vector_2 = [next_detection[0] - detection3[0], next_detection[1] - detection3[1]]
  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)
  if dot_product > 1:
    dot_product = 1
  elif dot_product < -1:
    dot_product = -1
  angle = np.arccos(dot_product) *  180 / np.pi 
  
  if angle <= 1:
    angle_score = np.log(1/180)
  else:
    angle_score = np.log(angle/180)
  angle_score = angle_score + 1.4# 1.4 =(- np.log(45/180))
  actual_score = actual_score + angle_score
  actual_score = max(actual_score,MIN_SCORE)

  return actual_score  
  

def get_prediction(detection1, detection2, detection3, multiplicator = 1): 
  """
    Description:
      Compute predtiction based on motion model from last 3 detections in trajectory 
    Parameters:
      detection1     (tuple[int,int]): coordinates x-y of third detection from end of trajectory
      detection2     (tuple[int,int]): coordinates x-y of second detection from end of trajectory
      detection3     (tuple[int,int]): coordinates x-y of first detection from end of trajectory
      multiplicator  (int)           : number of frames the next_detection is ahead of detection3. 
    Returns:
                     (int,int)       : x-y coordinates of prediction
  """

  acceleration_x = ( (detection3[0] - detection2[0]) - (detection2[0] - detection1[0]) ) / (DELTA_T*DELTA_T)
  acceleration_y = ( (detection3[1] - detection2[1]) - (detection2[1] - detection1[1]) ) / (DELTA_T*DELTA_T)

  acceleration_x_last_two = (detection3[0] - detection2[0])
  acceleration_y_last_two = (detection3[1] - detection2[1])
  if (acceleration_x_last_two > 0 and acceleration_x < 0) or (acceleration_x_last_two < 0 and acceleration_x > 0):
    acceleration_x = 0
  if (acceleration_y_last_two > 0 and acceleration_y < 0) or (acceleration_y_last_two < 0 and acceleration_y > 0):
    acceleration_y = 0

  velocity_x = ( ( (detection3[0] - detection2[0]) ) / DELTA_T ) + acceleration_x * DELTA_T
  velocity_y = ( ( (detection3[1] - detection2[1]) ) / DELTA_T ) + acceleration_y * DELTA_T
  prediction_x = detection3[0] + ( velocity_x * DELTA_T + ( (acceleration_x * DELTA_T*DELTA_T) / 2 ) ) * multiplicator
  prediction_y = detection3[1] + (velocity_y * DELTA_T + ( (acceleration_y * DELTA_T*DELTA_T) / 2 ) )* multiplicator
  return (prediction_x,prediction_y)  



def is_spatial_overlap(path1, path2): 
  """
    Description:
      Decides whether two trajectories has spatial overlap. Whether they share detections on same or overlaping frames and if so if the detections are the same. 
      If yes, then returns True, otherwise returns False. 
      Order of the paths matters, path1 should precede path2.
    Parameters:
      path1                        (tuple[int,int,int]): list of detections of trajectory in format (frame num, x, y). Sorted by frame num.
      path2                        (tuple[int,int,int]): list of detections of trajectory in format (frame num, x, y). Sorted by frame num.
    Returns:
      same_detections_in_overlap    (bool)             : True if two trajectories spatialy overlap. False otherwise.
  """

  if path1[-1][0] > path2[-1][0]: #assuming that path1 is ending sooner than path2
    return False
  if (path2[-1][0] >= path1[-1][0] and path2[0][0] <= path1[0][0]) or (path2[-1][0] <= path1[-1][0] and path2[0][0] >= path1[0][0]):#paths are same or contained in each other with respect to the frame sequence
    return False

  match = False
  match_id = 0
  #looking for frame overlap
  for i in range(len(path1)-1,-1,-1):
    if path1[i] == path2[0]:
      match = True
      match_id = i
      break

  same_detections_in_overlap = False
  if match:
    same_detections_in_overlap = True
    for i in range(match_id,len(path1)): 
        if path1[i] != path2[i-match_id]:
          same_detections_in_overlap = False
          break

  return same_detections_in_overlap



def get_unit_vector_from_prediction(detection1, detection2, detection3):
  """
    Description:
      Computes unit vector based on vector between detection3 and predction made from detections1-3. Vector points from detection3 to prediction.

    Parameters:
      detection1    (tuple(int,int))    : coordinates x-y of third detection from end of trajectory
      detection2    (tuple(int,int))    : coordinates x-y of second detection from end of trajectory
      detection3    (tuple(int,int))    : coordinates x-y of first detection from end of trajectory
    Returns:
      unit_vector_1 (tuple[float,float]): unit vector between detection3 and predction
  """

  prediction_x, prediction_y = get_prediction(detection1,detection2,detection3)
  vector_1 = [prediction_x - detection3[0], prediction_y - detection3[1]]
  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  return unit_vector_1


def merge_paths_spatially(path1, path2): 
  """
    Description:
      Merge two paths that overlaps spatialy. Function "is_spatial_overlap" should return true.
    Parameters:
      path1           (tuple[int,int,int]): list of detections of trajectory in format (frame num, x, y). Sorted by frame num.
      path2           (tuple[int,int,int]): list of detections of trajectory in format (frame num, x, y). Sorted by frame num.
    Returns:
      paths_merged    (tuple[int,int,int]): path1 nad path2 merged based on spatial overlap
  """

  if path1 == []:
    return path2
  elif path2 == []:
    return path1
  
  if path1.path_coords[-1][0] > path2.path_coords[-1][0]: #path1 should precede path2, otherwise switch them
    tmp = path2
    path2 = path1
    path1 = tmp

  last_index_of_path1 = path1.path_coords[-1][0]
  intersection_end_index = 'x' #default value signalizing error

  for i in range(len(path2.path_coords)):
    if path2.path_coords[i][0] == last_index_of_path1:
      intersection_end_index = i
      break

  if intersection_end_index == 'x': #paths do not spatialy intersect
    print(f"error merge spatial {path1.path_coords} and {path2.path_coords}")
    raise ValueError  

  score = path1.score + path2.score - np.sum(path2.score_arr[0:intersection_end_index+1]) 
  path = path1.path + path2.path[intersection_end_index + 1:] 
  path_coords = path1.path_coords + path2.path_coords[intersection_end_index + 1:]
  predictions = path1.predictions + path2.path_coords[intersection_end_index + 1:]
  score_arr = path1.score_arr + path2.score_arr[intersection_end_index + 1:]


  paths_merged = Path([],[],[])
  paths_merged.path = path
  paths_merged.path_coords = path_coords
  paths_merged.predictions = predictions
  paths_merged.score = score
  paths_merged.score_arr = score_arr

  return paths_merged

def should_merge_by_interpolation(path1, path2, max_y_diff_for_merge):
  """
    Description:
      Decides whether two trajectories that do not overlap spatialy should be merged by interpolation. 
      Decision is made by getting vector between prediction of next detection from last/first three detections in path1/2 and last/first detectin in path1/2.
      Vector is rotated by +/- angle. 
      If lines in directions of that vectors overlap, then distance between two paths is computed. 
      If distance is smaller than threshold, return True, otherwise False.
    Parameters:
      path1        (tuple[int,int,int]): list of detections of trajectory in format (frame num, x, y). Sorted by frame num.
      path2        (tuple[int,int,int]): list of detections of trajectory in format (frame num, x, y). Sorted by frame num.
    Returns:
      should_merge (bool)              : whether paths should be merged by interpolation.
  """

  if path2.path_coords[0][0] < path1.path_coords[-1][0]: #path1 should precede path2, otherwise false
    return False

  #lines will have format of two end points
  lines_path1 = []
  lines_path2 = []
  rotations = [0,30,330] #rotation made

  
  #last three detections of path1
  detection1_1 = (path1.path_coords[-3][1],path1.path_coords[-3][2] )
  detection1_2 = (path1.path_coords[-2][1],path1.path_coords[-2][2] )
  detection1_3 = (path1.path_coords[-1][1],path1.path_coords[-1][2] )
  #first three detections of path2
  detection2_1 = (path2.path_coords[2][1],path2.path_coords[2][2] )
  detection2_2 = (path2.path_coords[1][1],path2.path_coords[1][2] )
  detection2_3 = (path2.path_coords[0][1],path2.path_coords[0][2] )


  unit_vector_1 = get_unit_vector_from_prediction(detection1_1,detection1_2,detection1_3)
  unit_vector_2 = get_unit_vector_from_prediction(detection2_1,detection2_2,detection2_3)

  for rotation in rotations:
    unit_vector_1_rotated = rotate_vector(unit_vector_1,rotation)
    end_point_rot1 = get_end_point_from_vector_and_point(unit_vector_1_rotated,detection1_3,widht,height)
    lines_path1.append((detection1_3,end_point_rot1))

    unit_vector_2_rotated = rotate_vector(unit_vector_2,rotation)
    end_point_rot2 = get_end_point_from_vector_and_point(unit_vector_2_rotated,detection2_3,widht,height)
    lines_path2.append((detection2_3,end_point_rot2))

  #find if exist any intersection between 3 lines from path1 and 3 lines from path2
  is_intersection = False
  for line1 in lines_path1:
    for line2 in lines_path2:
      px,py = get_intersection_of_two_lines_segments(line1[0],line1[1],line2[0],line2[1])
      if px != -1 and py != -1:
        is_intersection = True
        break
    if is_intersection:
      break

  frame_distance = path2.path_coords[0][0] - path1.path_coords[-1][0]

  should_merge = False
  dist = (compute_distance_abs(detection1_3, (px,py)) + compute_distance_abs(detection2_3, (px,py)))
  dist_y = abs(detection1_3[1] - py) + abs(detection2_3[1] - py)
  if dist < ( frame_distance * MAX_DIST ) and is_intersection and dist_y < max_y_diff_for_merge:
    should_merge = True

  return should_merge



def merge_paths_by_intersection(path1, path2):
  """
    Description:
      Merge two paths that do not overlap spatialy. But should be merged by interpolation
    Parameters:
      path1        (tuple[int,int,int]): list of detections of trajectory in format (frame num, x, y). Sorted by frame num.
      path2        (tuple[int,int,int]): list of detections of trajectory in format (frame num, x, y). Sorted by frame num.
    Returns:
      paths_merged (tuple[int,int,int]): path1 nad path2 merged. path2 append to path1
  """

  if path1 == []:
    return path2
  elif path2 == []:
    return path1
  
  if path1.path_coords[-1][0] > path2.path_coords[0][0]: #path1 should precede path2, otherwise switch them
    tmp = path2
    path2 = path1
    path1 = tmp
  path1_len = len(path1.path_coords)
  path = path1.path + path2.path
  path_coords = path1.path_coords + path2.path_coords
  score_arr = path1.score_arr + path2.score_arr
  predictions = path1.predictions + path2.predictions
  score = path1.score + path2.score 
  score1 = compute_score2((path_coords[path1_len-3][1],path_coords[path1_len-3][2]),(path_coords[path1_len-2][1],path_coords[path1_len-2][2]),(path_coords[path1_len-1][1],path_coords[path1_len-1][2]),(path_coords[path1_len][1],path_coords[path1_len][2]))
  score2 = compute_score2((path_coords[path1_len-2][1],path_coords[path1_len-2][2]),(path_coords[path1_len-1][1],path_coords[path1_len-1][2]),(path_coords[path1_len][1],path_coords[path1_len][2]),(path_coords[path1_len+1][1],path_coords[path1_len+1][2]))
  score3 = compute_score2((path_coords[path1_len-1][1],path_coords[path1_len-1][2]),(path_coords[path1_len][1],path_coords[path1_len][2]),(path_coords[path1_len+1][1],path_coords[path1_len+1][2]),(path_coords[path1_len+2][1],path_coords[path1_len+2][2]))

  score = score + score1 + score2 + score3

  paths_merged = Path([],[],[])
  paths_merged.path = path
  paths_merged.path_coords = path_coords
  paths_merged.predictions = predictions
  paths_merged.score = score
  paths_merged.score_arr = score_arr

  return paths_merged


def get_intersection_of_two_lines_segments(start1,end1,start2,end2):
  """
    Description:
      Find intersection point of two lines (start1,end1), (start2,end2).
      https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    Parameters:
      start1       (tuple(int,int)): first end point of line1
      end1         (tuple(int,int)): second end point of line1
      start2       (tuple(int,int)): first end point of line2
      end2         (tuple(int,int)): second end point of line2
    Returns:
      (px,py)      (tuple[int,int]): coordinates x-y of intersection point of two lines
  """

  x1,y1 = start1
  x2,y2 = end1
  x3,y3 = start2
  x4,y4 = end2
  D = ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
  if D == 0:
    return (-1,-1)
  px = ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / D
  py = ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / D
 
  if not ((((px >= x1 and px <= x2) or (px <= x1 and px >= x2)) and ((py >= y1 and py <= y2) or (py <= y1 and py >= y2))) and (((px >= x3 and px <= x4) or (px <= x3 and px >= x4)) and ((py >= y3 and py <= y4) or (py <= y3 and py >= y4)))): 
    px = -1
    py = -1
  return (px,py)  


def merge_paths_by_graph(option, window, tok_id, acc, graph, paths_per_windows): 
  """
    Description:
        DFS type algorithm that merges paths spatially or by intersection. Merging is done recursively by information from graph parameter that links paths
        by one of the merging option or none.
        Merges paths in one window.
    Parameters:
       option             (int):                              merging option, 0 - merge spatialy, 1 - merge by intersection
       window             (int):                              window id of current path
       tok_id             (int):                              id of current path in window
       acc                (list(Path)):                       accumulator of paths made by merging process
       graph              (list(list(tuple[int,int,int]))):   list of lists of tuples representing edges between paths with merging option, Format of tuple (option, path_id of first patj, path_id of second path)
       paths_per_windows  (list(list[Path])):                 list of lists of Path objects representing paths found in windows. Indexing paths_per_windows[window id][path in window id].
    Returns:
       acc (list(Path)): list of paths made by merging process
  """

  if option == 0:
    acc = merge_paths_spatially(acc,paths_per_windows[window][tok_id])
  elif option == 1:
    acc = merge_paths_by_intersection(acc,paths_per_windows[window][tok_id])

  acc_tmp = copy.deepcopy(acc)
  acc_tmp2 = copy.deepcopy(acc)
  acc = []

  for i in range(len(graph[tok_id])):
    for path in merge_paths_by_graph(graph[tok_id][i][0],window,graph[tok_id][i][2],acc_tmp,graph,paths_per_windows):
      acc.append(path)
      acc_tmp = copy.deepcopy(acc_tmp2)

  if len(graph[tok_id]) == 0:
      acc = [copy.deepcopy(acc_tmp)] 

  return acc  

def make_windows(windows_size, window_shift, detections):
  """
    Description:
      Divide detections in frame into windows of size "windows_size" and shift between them of "window_shift"
    Parameters:
      windows_size                      (int)                                               : size of windows
      window_shift                      (int)                                               : shift of windows
      detections                        (list(list(tuple[int,int])))                        : list of lists of detections in frames
    Returns:
      (windows, start_frame_of_windows) (tuple[list(list(list(tuple[int,int]))),list(int)]) : windows: is list of lists of lists of detections per window per frame, indexing [window][frame][detection], start_frame_of_windows: is list of number of starting frame in windows
  """
  windows = []
  start_frame_of_windows = []
  i = 0
  num_frames = len(detections)
  while True:
    left = window_shift*i 
    right = left + windows_size
    if right > num_frames:
      break
    window = detections[left:right]
    windows.append(window)
    start_frame_of_windows.append(left)
    i = i + 1
    
  if (window_shift*(i-1) + windows_size) < num_frames: #uncomplete window at the end
    left = window_shift * (i)
    right = num_frames
    window = detections[left:]
    windows.append(window)
    start_frame_of_windows.append(left)

  return (windows, start_frame_of_windows)

def make_connection_graph_per_window(windows):
  """
    Description:
      Make graph of connection between detections in window. Edge from detection A to detection B is made if A is on frame i and B on frame i+1 and distance is between 
      MAX_DIST and MIN_DIST.
    Parameters:
      windows                        (list(list(list(tuple[int,int])))) : list of lists of lists of detections per window per frame. indexing [window][frame][detection]
    Returns:
      graph_edges_indexes_per_window (list(list(list(int))))            : list of lists of lists indexed by [window][frame][detection] containing edges, see desc.
  """ 
  graph_edges_indexes_per_window = []
  for window in windows:
    detections_per_frames = window
    num_detections = len(detections_per_frames) 

    graph_edges_indexes = []
    for i in range(num_detections): #each frame a list
      graph_edges_indexes.append([])

    for i in range(len(detections_per_frames)-1): #frames in window, detections in last frame do not have succesors
      for j in range(len(detections_per_frames[i])): #detections in frame i
        tmp_list = [] #list of successors for detection j in frame i+1
        for k in range(len(detections_per_frames[i+1])): #detections in frame i+1
          dist = compute_distance_abs(detections_per_frames[i][j],detections_per_frames[i+1][k])
          if dist <= MAX_DIST and dist >= MIN_DIST:
            tmp_list.append(k)
        graph_edges_indexes[i].append(tmp_list)
    graph_edges_indexes_per_window.append(graph_edges_indexes)

  return graph_edges_indexes_per_window

def compare_paths(path1, path2): #weighted by length of path, penalizing paths with shorter length, because score is made from fourth detections in path 
  """
    Description:
      Compares two paths by comparing their score divided by length
    Parameters:
      path1       (Path) : 
      path2       (Path) :
    Returns:
                  (int) : positive if path1 is better, negative if path2 is better, 0 if they are same
  """
  return (path1.score / (len(path1.path_coords))) - (path2.score/ (len(path2.path_coords)))

def compare_paths_absolute(path1, path2):
  """
  Description:
    Compares two paths by comparing their score
  Parameters:
    path1       (Path) : 
    path2       (Path) : 
  Returns:
                (int)  : positive if path1 is better, negative if path2 is better, 0 if they are same
  """
  return (path1.score) - (path2.score)

def compare_paths_by_first_frame_order(path1, path2):
  """
    Description:
      Compares two paths by comparing their first frame order number
    Parameters:
      path1       (Path) : 
      path2       (Path) : 
    Returns:
                  (int)  : positive if path1 first frame order number is higher, negative if path2 first frame order number is higher, 0 if first frame order numbers are same
  """
  return path1.path_coords[-1][0] - path2.path_coords[-1][0]
    
def update_path(path, next_detection, next_detection_index_of_frame_in_window, next_detection_index_of_detection_in_frame, next_detection_index_of_frame_global):
  """
    Description:
      Updates path by adding new detection. Computes score for the path as well.
    Parameters:
      path                                        (Path)            : Path object
      next_detection                              (tuple[int,int])  : detection to be added in format (x,y)
      next_detection_index_of_frame_in_window     (int)             : index of frame in window, first frame in window is 0
      next_detection_index_of_detection_in_frame  (int)             : index of detection in frame, detections are grouped by frames
      next_detection_index_of_frame_global        (int)             : index of frame in global view, not in window view
    Returns:
      path                                        (Path)            : updated Path by new detection
  """  
  if len(path.path_coords) >= 3: #if path has at least 3 members, compute score
    if path.score_without_shifts_weightings == float("inf"): 
        path.score_without_shifts_weightings = 0 
    multiplicator = max(path.virtual_nodes_counter,1)
    actual_score = compute_score(path, next_detection,multiplicator)
    path.score_arr.append(actual_score)
    path.score_without_shifts_weightings = path.score_without_shifts_weightings + actual_score

    detection1 = (path.path_coords[-2][1],path.path_coords[-2][2])
    detection2 = (path.path_coords[-1][1],path.path_coords[-1][2])
    angle = get_angle2(detection1,detection2,next_detection)
    if angle >= 70:#to big difference between prediction of next detection and next detection
      path.cnt_dir_shifts = path.cnt_dir_shifts +1

    if path.cnt_dir_shifts == 0:
        path.score = path.score_without_shifts_weightings
    elif path.score_without_shifts_weightings >= 0:
        path.score = path.score_without_shifts_weightings  * (1+(path.cnt_dir_shifts / (len(path.path_coords)+1)))
    else:  
      path.score = path.score_without_shifts_weightings  * (1-(path.cnt_dir_shifts / (len(path.path_coords)+1)))
    

  path.path.append((next_detection_index_of_frame_in_window,next_detection_index_of_detection_in_frame)) #append frame num and index of detection in next frame
  path.path_coords.append((next_detection_index_of_frame_global,next_detection[0],next_detection[1])) #frame,x,y
  path.virtual_nodes_counter = 0
  path.x_movement = path.x_movement + abs(path.path_coords[-1][1] - path.path_coords[-2][1])
  path.y_movement = path.y_movement + abs(path.path_coords[-1][2] - path.path_coords[-2][2])
  return path



def find_paths_in_windows(windows, graph_edges_indexes_per_window, start_frame_of_windows):
  """
    Description:
      Find paths from detections inside windows. Algorithm 1 from paper.
    Parameters:
      windows                                 (list(list(list(tuple[int,int]))))  : list of lists of lists of detections per window per frame. indexing [window][frame][detection]
      graph_edges_indexes_per_window          (list(list(list(int))))             : list of lists of lists indexed by [window][frame][detection] containing edges of detection to other detections
      start_frame_of_windows                  (list(int))                         : prefix sum of number of frames in windows
    Returns:
      paths_per_windows                       (list(list(Path)))                  : paths founded in each window, indexing [window][path]
  """
  paths_current = []
  paths_next = []
  paths_ended = []
  paths_per_windows = []
  win_iter = -1
  for window,graph_edges_indexes,index_of_first_frame_in_window in zip(windows,graph_edges_indexes_per_window,start_frame_of_windows):
    win_iter = win_iter + 1
    num_frames = len(window)
    detections = window
    detections_in_trajectories_indexes_current = [] #set
    paths_next = []
    paths_ended = []
    paths_current = []

    for i in range(num_frames-1):#for all frames, line 2
      if len(paths_next) > paths_to_keep_to_next_round and i >= 3: 
        paths_current = []
        paths_next = sorted(paths_next,  key=functools.cmp_to_key(compare_paths))

        for e in range(len(paths_next)-1,-1,-1):
          if len(paths_next[e].path_coords) > 3 and x_vs_y_movement_threshold*(paths_next[e].x_movement) > paths_next[e].y_movement:
            paths_next.pop(e)

        best_score = paths_next[0].score / (len(paths_next[0].path_coords))
        
        for idx in range(len(paths_next)): 
          if (paths_next[idx].score / (len(paths_next[idx].path_coords))) == best_score:
            paths_current.append(paths_next[idx])
            paths_ended.append(paths_next[idx])
          elif idx < paths_to_keep_to_next_round:
            paths_current.append(paths_next[idx])
            if idx < 4:
              paths_ended.append(paths_next[idx])
          elif len(paths_next[idx].path) < 4:
            paths_current.append(paths_next[idx])
          else:
            break

      else:
        paths_current = paths_next
      
 
      paths_next = []#line 9
      detections_in_trajectories_indexes_current = []
      for path in paths_current:
        if path.path[-1][0] == i:
          detections_in_trajectories_indexes_current.append(path.path[-1][1])

      for j in range(len(detections[i])): #line 2 for all detections in frame
          paths_current.append(Path((i,j),(index_of_first_frame_in_window+i,detections[i][j][0],detections[i][j][1]))) #frame i detection j

        #line 6,7 TODO
      for path in paths_current:#line 10    
        if len(graph_edges_indexes[path.path[-1][0]][path.path[-1][1]]) == 0 or path.virtual_nodes_counter > 0: #path cannot continue, can happen only limit_VCs times
          if path.virtual_nodes_counter < limit_VCs:
              
              actual_detection = path.path_coords[-1][1],path.path_coords[-1][2] 
              path.virtual_nodes_counter = path.virtual_nodes_counter + 1 
              succes = False

              for l in range(len(detections[i+1])):#all detections in next frame
                next_detection = detections[i+1][l]
                # make bigger acceptable distance for next attempt
                if compute_distance_abs(actual_detection, next_detection) < (MAX_DIST * (1+ (path.virtual_nodes_counter/100))) and compute_distance_abs(actual_detection, next_detection) >= MIN_DIST: #dostatecne mala vzdalenost
                  path_tmp = copy.deepcopy(path)
                  succes = True
                  path_tmp = update_path(path_tmp,next_detection,i+1,l,index_of_first_frame_in_window + i+1) ###########
                  paths_next.append(path_tmp) #continue, next detection not added, but number of misses in limit_VCs limit
              if not succes: #path ending, cant add detection to path for limit_VCs times
                paths_next.append(path)
          else: 
            if len(path.path) > 3:
              paths_ended.append(path)
        else:
          path.virtual_nodes_counter = 0
          for k in range(len(graph_edges_indexes[path.path[-1][0]][path.path[-1][1]])): #edges for last detection in path
            next_detection = detections[i+1][graph_edges_indexes[path.path[-1][0]][path.path[-1][1]][k]]#line 11, detection[#frame][xx] where xx is graph_edges_indexes[#frame][#detection on frame][neighbor id]
            #line 12,13,14
            path_tmp = copy.deepcopy(path) 
            path_tmp = update_path(path_tmp,next_detection,i+1,graph_edges_indexes[path.path[-1][0]][path.path[-1][1]][k],index_of_first_frame_in_window + i+1)#########
            paths_next.append(path_tmp) #line 15, next detection to path added continue to next iteration

            
    #window iteration ended
    paths_next = sorted(paths_next,  key=functools.cmp_to_key(compare_paths))
    for path in paths_next[0:4]: #keep some paths builded to this iteration and append them to already stored paths in paths_ended
      if len(path.path_coords) >= 4:
        paths_ended.append(path) 
    for i in range(len(paths_ended)-1,-1,-1):
      if len(paths_ended[i].path_coords) < 4: #pop paths with length smaller than 4 detections, because score cant be computed for them
        paths_ended.pop(i)
    paths_ended = sorted(paths_ended,  key=functools.cmp_to_key(compare_paths_absolute)) 
    paths_per_windows.append(paths_ended)

  return paths_per_windows  



def merge_paths_inside_windows(windows, paths_per_windows, max_y_diff_for_merge):
  """
    Description:
      Merge paths inside windows. Merge if temporarily + spatially overlaps or do not overlap but their intersection has short enough distance. 
      Intersection is found when lines in direction of prediction from end of first path and start of second path intersects. Lines are also rotated by small degree.
      Cases when to merge are described in paper. 
    Parameters:
      windows                     (list(list(list(tuple[int,int]))))  : list of lists of lists of detections per window per frame. indexing [window][frame][detection]
      paths_per_windows           (list(list(Path)))                  : paths founded in each window, indexing [window][path]
    Returns:
      paths_per_windows           (list(list(Path)))                  : paths founded in each window plus paths merged inside windows, indexing [window][path]
  """
  
  for window_idx in range(len(windows)):
    paths_per_windows[window_idx] = sorted(paths_per_windows[window_idx],  key=functools.cmp_to_key(compare_paths_by_first_frame_order)) #aby pathy sli casove po sobe podle posledniho framu, jinak se nechova korektne
    #for each path in each window make list for edges
    graph_paths_edges_indexes = []
    for i in range(len(paths_per_windows[window_idx])):
        graph_paths_edges_indexes.append([])


    for k in range(len(paths_per_windows[window_idx])):# from path
      for l in range(k+1,len(paths_per_windows[window_idx])): #to path
        path1 = paths_per_windows[window_idx][k]
        path2 = paths_per_windows[window_idx][l]
        if path1.path_coords[-1][0] >= path2.path_coords[0][0] and path1.path_coords[0][0] <= path2.path_coords[-1][0]:
          if is_spatial_overlap(path1.path_coords,path2.path_coords):
            graph_paths_edges_indexes[k].append((0,k,l))#option/case,from window,path to window,path 
        else:
          should_merge = should_merge_by_interpolation(path1,path2, max_y_diff_for_merge)
          if should_merge:
            graph_paths_edges_indexes[k].append((1,k,l))
    new_paths = []
    #build new paths inside windows
    for k in range(len(paths_per_windows[window_idx])):
      if len(graph_paths_edges_indexes[k]) > 0:
        new_paths = new_paths + merge_paths_by_graph(0,window_idx,k,[],graph_paths_edges_indexes,paths_per_windows)
    new_paths = sorted(new_paths,  key=functools.cmp_to_key(compare_paths)) #zbytecne, ale pro debug

    for path in new_paths[0:4]:
      paths_per_windows[window_idx].append(path)

  return paths_per_windows

def reduce_number_of_paths_in_windows(paths_per_windows, upper_limit):
  """
    Description:
      Reduce number of paths in each window to maximum of size upper_limit.
    Parameters:
      paths_per_windows       (list(list(Path))) : paths founded in each window, indexing [window][path]
      upper_limit             (list(list(Path))) : paths founded in each window, indexing [window][path]
    Returns:
      paths_per_windows       (list(list(Path))) : paths_per_windows with reduced size of paths in each window
  """
  for window_idx in range(len(paths_per_windows)): 
    paths_per_windows[window_idx] = sorted(paths_per_windows[window_idx],  key=functools.cmp_to_key(compare_paths)) #zbytecne, ale pro debug
    paths_per_windows[window_idx] = paths_per_windows[window_idx][0:upper_limit]
  return paths_per_windows


def make_graph_for_paths_in_different_windows(windows, paths_per_windows, max_windows_forward, max_y_diff_for_merge):
  """
    Description:
      Connects paths in between windows. Path in window i can be connected to path in window i+1...i+max_windows_forward. 
      Connect if temporarily + spatially overlaps or do not overlap but their intersection has short enough distance. 
      Intersection is found when lines in direction of prediction from end of first path and start of second path intersects. Lines are also rotated by small degree.
      Cases when to merge are described in paper. 
    Parameters:
      windows                               (list(list(list(tuple[int,int]))))        : list of lists of lists of detections per window per frame. indexing [window][frame][detection]
      paths_per_windows                     (list(list(Path)))                        : paths founded in each window, indexing [window][path]
      max_windows_forward                   (int)                                     : max windows forward from which paths can be connected to paths from current window
    Returns:
      graph_paths_edges_indexes_per_window  (list(list(tupple[int,int,int,int,int]))) : connections between paths, see desc. indexing [window][path], 
                                                                                        tupple structure[connection option, from window, from path, to window, to path]
  """
  
  graph_paths_edges_indexes_per_window = []
  for paths in paths_per_windows:
    num_paths = len(paths)
    graph_paths_edges_indexes = []
    for i in range(num_paths):
      graph_paths_edges_indexes.append([])
    graph_paths_edges_indexes_per_window.append(graph_paths_edges_indexes)


  number_of_windows = len(paths_per_windows)
  for i in range(number_of_windows):

    for j in range(1,min(max_windows_forward+1, number_of_windows - i)):
      for k in range(len(paths_per_windows[i])):
        for l in range(len(paths_per_windows[i+j])):
          path1 = paths_per_windows[i][k]
          path2 = paths_per_windows[i+j][l]
          if path1.path_coords[-1][0] >= path2.path_coords[0][0] and path1.path_coords[0][0] <= path2.path_coords[-1][0]: 
            if is_spatial_overlap(path1.path_coords,path2.path_coords):
                graph_paths_edges_indexes_per_window[i][k].append((0,i,k,j+i,l))#option/case,from window,path to window,path 

          else:
            should_merge = should_merge_by_interpolation(path1,path2, max_y_diff_for_merge)
            if should_merge:
              graph_paths_edges_indexes_per_window[i][k].append((1,i,k,j+i,l))
  return graph_paths_edges_indexes_per_window  


def merge_paths_by_indexes(paths_indexes, partial_sum, connection_type, paths_per_windows):
  """
    Description:
      Merge paths. Paths to be merged are indicated in path parameter by indexes to sum of all pathhs that can be transformed to window_id-path_id by 
      partial_sum.
    Parameters:
      paths_indexes           (list(int))                 : indexes of paths to be merged
      partial_sum             (list(int))                 : prefix sum of number of paths in windows
      connection_type         (numpy.array(numpy.array))  : information of merge type between paths
      paths_per_windows       (list(list[Path]))          : paths founded in each window, indexing [window][path]
    Returns:
      merged_path             (Path)                      : path made by merges of paths
  """
  
  window_path_indexes = [] #window,tok
  for index in paths_indexes:
    window_path_index = split_index_to_window_and_path(index,partial_sum)
    window_path_indexes.append(window_path_index)

  merged_path = paths_per_windows[window_path_indexes[0][0]][window_path_indexes[0][1]] #first path
  for i in range(1,len(paths_indexes)):
    last_id = paths_indexes[i-1] #previous id
    to_merge_id = paths_indexes[i] #next id
    option = connection_type[last_id][to_merge_id]
    if option == 0:
      merged_path = merge_paths_spatially(merged_path,paths_per_windows[window_path_indexes[i][0]][window_path_indexes[i][1]])
    elif option == 1:
      merged_path = merge_paths_by_intersection(merged_path,paths_per_windows[window_path_indexes[i][0]][window_path_indexes[i][1]])

  return merged_path

  
def is_duplicate(path1, path2): 
  """
    Description:
      Checks if two paths are same.
    Parameters:
      path1     (list(int,int,int)) : frame,x,y
      path2     (list(int,int,int)) : frame,x,y
    Returns:
                (bool)      : True if they are same, False otherwise
  """
  if len(path1) != len(path2):
    return False
  for i in range(len(path1)):
    if path1[i] != path2[i]:
      return False
  return True
  
def remove_duplicites_in_two_windows(window1, window2):
  """
    Description:
      Remove paths from window2 if they are also in window1.
    Parameters:
      window1             (list(list(tuple[int,int])))  : list of lists of detections per frame. indexing [frame][detection]
      window2             (list(list(tuple[int,int])))  : list of lists of detections per frame. indexing [frame][detection]   
    Returns:
                          (list(list(tuple[int,int])))  : returns window2 without duplicities with window1
  """
  to_delete_indexes = []
  for i in range(len(window2)-1,-1,-1):
    for j in range(len(window1)):
      if is_duplicate(window1[j].path_coords,window2[i].path_coords):
        window2.pop(i)
        break
  return window2


def remove_duplicites_from_windows(paths_per_windows, windows_size, window_shift):
  """
    Description:
      Remove duplicate paths across windows.
    Parameters:
      paths_per_windows       (list(list[Path]))            : paths founded in each window, indexing [window][path]
      window1                 (list(list(tuple[int,int])))  : list of lists of detections per frame. indexing [frame][detection]
      window2                 (list(list(tuple[int,int])))  : list of lists of detections per frame. indexing [frame][detection]   
    Returns:
     paths_per_windows        (list(list[Path]))            : returns paths_per_windows  without duplicites across windows
  """
  MAX_WINDOWS_FORWARD = math.floor(windows_size / window_shift) - 1 
  number_of_windows = len(paths_per_windows)
  for i in range(number_of_windows):
    for j in range(1,min(MAX_WINDOWS_FORWARD+1, number_of_windows - i)):
      paths_per_windows[j+i] = remove_duplicites_in_two_windows(paths_per_windows[i], paths_per_windows[j+i])
  return paths_per_windows
  
def check_if_in_base_lines_zone(path, court_points_per_image):
  """
    Description:
      Checks if path is inside .horizontal base line zone
    Parameters:
      path                        (list(int,int,int))       : frame,x,y
      corner_points_per_image     (list(tuple(int,int)))    : court points found in image             
    Returns:
                (bool)      : True if whole path inside horizontal base line zone
  """
  in_top_line_area = True
  in_bottom_line_area = True
  for i in range(len(path)):
    
    frame = path[i][0]
    if court_points_per_image[frame] is None:
      return False # missing court detection, cant decide
    left_top_corner = court_points_per_image[frame][13]
    right_top_corner = court_points_per_image[frame][10]
    left_bottom_corner = court_points_per_image[frame][3]
    right_bottom_corner = court_points_per_image[frame][0]
    detection_x = path[i][1]
    detection_y = path[i][2]
    #check top
    max_y = max(left_top_corner[1],right_top_corner[1])
    min_y = min(left_top_corner[1],right_top_corner[1])
    max_x = max(left_top_corner[0],right_top_corner[0])
    min_x = min(left_top_corner[0],right_top_corner[0])
    if not ((max_x >= detection_x >= min_x) and ((max_y+LINE_SIZE) >= detection_y >= (min_y-LINE_SIZE))):
      in_top_line_area = False

    max_y = max(left_bottom_corner[1],right_bottom_corner[1])
    min_y = min(left_bottom_corner[1],right_bottom_corner[1])
    max_x = max(left_bottom_corner[0],right_bottom_corner[0])
    min_x = min(left_bottom_corner[0],right_bottom_corner[0])
    if not ((max_x >= detection_x >= min_x) and ((max_y+LINE_SIZE) >= detection_y >= (min_y-LINE_SIZE))):
      in_bottom_line_area = False

  return in_bottom_line_area or in_top_line_area
  
  
def remove_paths_in_base_line_areas(paths_per_windows, court_points_per_image):
  """
    Description:
      Remove paths that are in.horizontal base line zone
    Parameters:
      paths_per_windows             (list(list[Path]))        : paths founded in each window, indexing [window][path]
      court_points_per_image        (list(tuple(int,int)))    : court points found in image             
    Returns:
      paths_per_windows             (list(list[Path]))        : paths_per_windows without removed paths
  """

  for i in range(len(paths_per_windows)):
    for j in range(len(paths_per_windows[i])-1,-1,-1):
      if check_if_in_base_lines_zone(paths_per_windows[i][j].path_coords, court_points_per_image):
        paths_per_windows[i].pop(j)
  return paths_per_windows
  
  
 
def constructPath(u, v, Next):     
    #implementation of floyd warshall from https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/
    # If there's no path between
    # node u and v, simply return
    # an empty array
    if (Next[u][v] == -1):
        return {}

    # Storing the path in a vector
    path = [u]
    while (u != v):
        u = Next[u][v]
        path.append(u)
 
    return path

def split_index_to_window_and_path(index, partial_sum):
  """
    Description:
      From index and partial sum of number of paths inside windows get window id and path id inside that window
    Parameters:
      index               (int)       : index of path in all paths inside one list
      partial_sum         (list(int)) : partial sum of number of paths in windows          
    Returns:
                          ((int,int)) : window id and path id inside that window
  """
  for i in range(len(partial_sum)-1,-1,-1):
    if partial_sum[i] <= index:
      return (i,index - partial_sum[i])
      
      
def find_shortest_lightest_trajectory_floyd_warshall_pycuda(paths_per_windows,graph_paths_edges_indexes_per_window):
  """
    Description:
      Find all disjoint trajectories from paths between windows. Use PyCUDA GPU parallel version of FW algorithm
    Parameters:
      paths_per_windows                       (list(list[Path]))                          : paths founded in each window, indexing [window][path]
      graph_paths_edges_indexes_per_window    (list(list(tupple[int,int,int,int,int])))   : connections between paths, see desc. indexing [window][path], 
                                                                                            tupple structure[connection option, from window, from path, to window, to path]
    Returns:
    paths_merged_lst                          (list(Path))                                : list of merged disjoint lines
    path_lst                                  (list(list(int)))                           : list of indexes of paths used per merged path, index is global to all paths from all windows                                                                                          get index window,path by function split_index_to_window_and_path
    partial_sum_arr                           (list(int))                                 : partial sum of number of paths in windows
  """
  number_of_windows = len(paths_per_windows)
  number_of_toks = 0
  partial_sum_arr = []
  for i in range(number_of_windows):
    partial_sum_arr.append(number_of_toks)
    number_of_toks = number_of_toks + len(paths_per_windows[i])
  dist = np.full((number_of_toks + 1, number_of_toks + 1), np.float32(INF)) #table of distances, paths go window by window and path by path inside window
  Next = np.full((number_of_toks + 1, number_of_toks + 1), -1) 
  connection_type = np.full((number_of_toks + 1, number_of_toks + 1),-1) #table with type of connection


  for window_nodes in graph_paths_edges_indexes_per_window: 
    for node in window_nodes:
      for edge in node:
        from_window = edge[1]
        from_tok = edge[2]
        to_window = edge[3]
        to_tok = edge[4]
        #####
        
        option = edge[0]
        if option == 1:
          path1 = paths_per_windows[from_window][from_tok]
          path2 = paths_per_windows[to_window][to_tok]
          merged_path = merge_paths_by_intersection(path1,path2)
          score =  merged_path.score - path1.score
        else:
          path1 = paths_per_windows[from_window][from_tok]
          path2 = paths_per_windows[to_window][to_tok]
          merged_path = merge_paths_spatially(path1,path2)
          score =  merged_path.score - path1.score
        
        from_node_idx = from_tok + partial_sum_arr[from_window]
        to_node_idx = to_tok + partial_sum_arr[to_window]
        dist[from_node_idx][to_node_idx] = paths_per_windows[to_window][to_tok].score
        connection_type[from_node_idx][to_node_idx] = edge[0]
        
  for i in range(number_of_toks):
    dist[number_of_toks][i] = 0
  
  for window_idx in range(number_of_windows): #last row in FW matrix is artificial start node
    for k in range(len(paths_per_windows[window_idx])):
      offset = partial_sum_arr[window_idx]
      dist[number_of_toks][offset + k] = paths_per_windows[window_idx][k].score

  V = number_of_toks + 1
  for i in range(V):
    for j in range(V):
        # No edge between node
        # i and j
        if (dist[i][j] == np.float32(INF)):
            Next[i][j] = -1
        else:
            Next[i][j] = j
            

  # CUDA code for FW
  mod = SourceModule("""
  #include <stdio.h>

  __global__ void FW_inner_loop(float * dis, int * next, int k, int size, int INF)
  {
     //calculates unique thread ID in the block
     int tid = (blockDim.x*blockDim.y)*threadIdx.z + (threadIdx.y*blockDim.x) + (threadIdx.x); 
     
     //calculates unique block ID in the grid
     int bid = (gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x);
     
     //block size (this is redundant though)
     int block_size = blockDim.x*blockDim.y*blockDim.z;
     
     //grid size (this is redundant though)
     int grid_size = gridDim.x*gridDim.y*gridDim.z;
     
     /*
     * Each cell in the matrix is assigned to a different thread.
     * Each thread do O(number of asssigned cell) computation.
     * Assigned cells of different threads does not overlape with
     * each other. And so no need for synchronization.
     */
  for (int i=bid; i<size; i+=grid_size)
     {
        for(int j=tid; j<size; j+=block_size)
        {
           if (dis[i*size+k] != INF && dis[k*size+j] != INF)
           {
             if (dis[i*size+j] > (dis[i*size+k]+dis[k*size+j]))
             {
               dis[i*size+j] = (dis[i*size+k]+dis[k*size+j]);
               next[i*size+j] = next[i*size+k];
             }
           }
           
        }
     }
  }

  """)
  

  
  dist = dist.astype(np.float32)
  Next = Next.astype(np.int32)
  dist = dist.flatten(order='C')
  Next = Next.flatten(order='C')

  dist_gpu = cuda.mem_alloc(dist.nbytes)
  Next_gpu = cuda.mem_alloc(Next.nbytes)
  cuda.memcpy_htod(dist_gpu, dist)
  cuda.memcpy_htod(Next_gpu, Next)
  
  func = mod.get_function("FW_inner_loop")
  threads =256
  blocks = min(2048/256,math.ceil(V/threads))
  for k in range(V):#(int k = 0; k < size; k++)
    func(dist_gpu,Next_gpu,np.int32(k),np.int32(V),np.float32(INF), block=(threads,1,1),grid=(blocks, 1))
    cuda.Context.synchronize()
 
  cuda.memcpy_dtoh(dist, dist_gpu)
  dist = np.reshape(dist, (-1, V), order = 'C')              
  cuda.memcpy_dtoh(Next, Next_gpu)
  Next = np.reshape(Next, (-1, V), order = 'C')   


    
    
  from_node = len(dist) - 1
  dist[-1][-1] = float("inf") #cant go to artificial node
  path = []
  path_lst = []
  paths_merged_lst = []
  break1 = False
  break2 = False
  #find disjoint trajectories
  while True:
    if path == []:
      to_node = np.argmin(dist[-1]) #look for minimum from artificial start node
      path = constructPath(from_node,to_node,Next)
      dist[-1][to_node] = float("inf") #delete connection for choosed connection
      path_lst.append(path)
      paths_merged_lst.append(merge_paths_by_indexes(path_lst[-1][1:],partial_sum_arr,connection_type,paths_per_windows))
    else:
      if break1 == False:#look for trajectories after last used node
        last_previous_node = path_lst[-1][-1]
        if np.min(dist[-1][last_previous_node+1:]) == float("inf"):
          break1 = True
        else:
          to_node = last_previous_node + 1 + np.argmin(dist[-1][last_previous_node+1:]) #+1 because argmin start from 0
          path2 = constructPath(from_node,to_node, Next)
          dist[-1][to_node] = float("inf")  

          if path2[1] > path_lst[-1][-1]:
            path_lst.append(path2)
            paths_merged_lst.append(merge_paths_by_indexes(path_lst[-1][1:],partial_sum_arr,connection_type,paths_per_windows))
            path = path2

      if break2 == False: #look for trajectories before first used node
        first_previous_node = path_lst[0][1] #index of first path in first trajectory found
        if np.min(dist[-1][0:first_previous_node]) == float("inf"):
            break2 = True
        else:
          to_node =  np.argmin(dist[-1][0:first_previous_node]) #+1
          path2 = constructPath(from_node,to_node, Next)
          dist[-1][to_node] = float("inf")  
          path_lst.insert(0,path2)
          paths_merged_lst.insert(0,merge_paths_by_indexes(path_lst[0][1:],partial_sum_arr,connection_type,paths_per_windows))
          path = path2

      if break1 and break2:
          break

    if path_lst[-1][-1] >=  partial_sum_arr[-1]: # at last node 
      break1 = True
    if path_lst[0][1] <=  0: #at firt node 
      break2 = True


  return paths_merged_lst,path_lst,partial_sum_arr
  
 
def find_shortest_lightest_trajectory_floyd_warshall_c(paths_per_windows,graph_paths_edges_indexes_per_window):
  """
    Description:
      Find all disjoint trajectories from paths between windows. Use C shared library version of FW algorithm
      code more documented in find_shortest_lightest_trajectory_floyd_warshall_pycuda
    Parameters:
      paths_per_windows                       (list(list[Path]))                          : paths founded in each window, indexing [window][path]
      graph_paths_edges_indexes_per_window    (list(list(tupple[int,int,int,int,int])))   : connections between paths, see desc. indexing [window][path], 
                                                                                            tupple structure[connection option, from window, from path, to window, to path]
    Returns:
    paths_merged_lst                          (list(Path))                                : list of merged disjoint lines
    path_lst                                  (list(list(int)))                           : list of indexes of paths used per merged path, index is global to all paths from all windows                                                                                          get index window,path by function split_index_to_window_and_path
    partial_sum_arr                           (list(int))                                 : partial sum of number of paths in windows
  """
  number_of_windows = len(paths_per_windows)
  number_of_toks = 0
  partial_sum_arr = []
  for i in range(number_of_windows):
    partial_sum_arr.append(number_of_toks)
    number_of_toks = number_of_toks + len(paths_per_windows[i])
  dist = np.full((number_of_toks + 1, number_of_toks + 1), np.float32(INF)) 
  Next = np.full((number_of_toks + 1, number_of_toks + 1), -1) 
  connection_type = np.full((number_of_toks + 1, number_of_toks + 1),-1) 


  
  for window_nodes in graph_paths_edges_indexes_per_window: 
    for node in window_nodes:
      for edge in node:
        from_window = edge[1]
        from_tok = edge[2]
        to_window = edge[3]
        to_tok = edge[4]
        
        option = edge[0]
        if option == 1:
          path1 = paths_per_windows[from_window][from_tok]
          path2 = paths_per_windows[to_window][to_tok]
          merged_path = merge_paths_by_intersection(path1,path2)
          score =  merged_path.score - path1.score
        else:
          path1 = paths_per_windows[from_window][from_tok]
          path2 = paths_per_windows[to_window][to_tok]
          merged_path = merge_paths_spatially(path1,path2)
          score =  merged_path.score - path1.score
        
        from_node_idx = from_tok + partial_sum_arr[from_window]
        to_node_idx = to_tok + partial_sum_arr[to_window]
        dist[from_node_idx][to_node_idx] = score
        connection_type[from_node_idx][to_node_idx] = edge[0]
  for i in range(number_of_toks):
    dist[number_of_toks][i] = 0
  
  for window_idx in range(number_of_windows): #artificial start node
    for k in range(len(paths_per_windows[window_idx])):
      offset = partial_sum_arr[window_idx]
      dist[number_of_toks][offset + k] = paths_per_windows[window_idx][k].score
  
  
  V = number_of_toks + 1
  for i in range(V):
    for j in range(V):
        # No edge between node
        # i and j
        if (dist[i][j] == np.float32(INF)):
            Next[i][j] = -1
        else:
            Next[i][j] = j
      
  
  
  dist = dist.astype(np.float32)
  Next = Next.astype(np.int32)
  dist = dist.flatten(order='C')
  Next = Next.flatten(order='C')
  
  c_lib = ctypes.CDLL(PATH_TO_FW_SHARED_LIBRARY)
  c_lib.FW(dist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),Next.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),ctypes.c_int(V),ctypes.c_int(INF))          
        
  dist = np.reshape(dist, (-1, V), order = 'C')              
  Next = np.reshape(Next, (-1, V), order = 'C')   


     

          
    
  from_node = len(dist) - 1
  dist[-1][-1] = float("inf") 
  path = []
  path_lst = []
  paths_merged_lst = []
  break1 = False
  break2 = False
  while True:
    if path == []:
      to_node = np.argmin(dist[-1]) 
      path = constructPath(from_node,to_node,Next)
      dist[-1][to_node] = float("inf")
      path_lst.append(path)
      paths_merged_lst.append(merge_paths_by_indexes(path_lst[-1][1:],partial_sum_arr,connection_type,paths_per_windows))
    else:
      if break1 == False:#looking for trajectory after last node used
        last_previous_node = path_lst[-1][-1]
        if np.min(dist[-1][last_previous_node+1:]) == float("inf"):
          break1 = True
        else:
          to_node = last_previous_node + 1 + np.argmin(dist[-1][last_previous_node+1:]) 
          path2 = constructPath(from_node,to_node, Next)
          dist[-1][to_node] = float("inf")  

          if path2[1] > path_lst[-1][-1]:
            path_lst.append(path2)
            paths_merged_lst.append(merge_paths_by_indexes(path_lst[-1][1:],partial_sum_arr,connection_type,paths_per_windows))
            path = path2

      if break2 == False: #looking for trajectory before first node used
        first_previous_node = path_lst[0][1] 
        if np.min(dist[-1][0:first_previous_node]) == float("inf"):
            break2 = True
        else:
          to_node =  np.argmin(dist[-1][0:first_previous_node]) 
          path2 = constructPath(from_node,to_node, Next)
          dist[-1][to_node] = float("inf")  
          path_lst.insert(0,path2)
          paths_merged_lst.insert(0,merge_paths_by_indexes(path_lst[0][1:],partial_sum_arr,connection_type,paths_per_windows))
          path = path2

      if break1 and break2:
          break

    if path_lst[-1][-1] >=  partial_sum_arr[-1]:
      break1 = True
    if path_lst[0][1] <=  0:
      break2 = True


  return paths_merged_lst,path_lst,partial_sum_arr
  

def find_shortest_lightest_trajectory_floyd_warshall_python(paths_per_windows,graph_paths_edges_indexes_per_window):
  """
    Description:
      Find all disjoint trajectories from paths between windows. Use Python version of FW algorithm
      code more documented in find_shortest_lightest_trajectory_floyd_warshall_pycuda
    Parameters:
      paths_per_windows                       (list(list[Path]))                          : paths founded in each window, indexing [window][path]
      graph_paths_edges_indexes_per_window    (list(list(tupple[int,int,int,int,int])))   : connections between paths, see desc. indexing [window][path], 
                                                                                            tupple structure[connection option, from window, from path, to window, to path]
    Returns:
    paths_merged_lst                          (list(Path))                                : list of merged disjoint lines
    path_lst                                  (list(list(int)))                           : list of indexes of paths used per merged path, index is global to all paths from all windows                                                                                          get index window,path by function split_index_to_window_and_path
    partial_sum_arr                           (list(int))                                 : partial sum of number of paths in windows
  """
  number_of_windows = len(paths_per_windows)
  number_of_toks = 0
  partial_sum_arr = []
  for i in range(number_of_windows):
    partial_sum_arr.append(number_of_toks)
    number_of_toks = number_of_toks + len(paths_per_windows[i])
  dist = np.full((number_of_toks + 1, number_of_toks + 1), float('inf')) 
  Next = np.full((number_of_toks + 1, number_of_toks + 1), -1) 
  connection_type = np.full((number_of_toks + 1, number_of_toks + 1),-1) 


  for window_nodes in graph_paths_edges_indexes_per_window: 
    for node in window_nodes:
      for edge in node:
        from_window = edge[1]
        from_tok = edge[2]
        to_window = edge[3]
        to_tok = edge[4]
        
        option = edge[0]
        if option == 1:#type of merge
          path1 = paths_per_windows[from_window][from_tok]
          path2 = paths_per_windows[to_window][to_tok]
          merged_path = merge_paths_by_intersection(path1,path2)
          score =  merged_path.score - path1.score
        else:
          path1 = paths_per_windows[from_window][from_tok]
          path2 = paths_per_windows[to_window][to_tok]
          merged_path = merge_paths_spatially(path1,path2)
          score =  merged_path.score - path1.score
        
        from_node_idx = from_tok + partial_sum_arr[from_window]
        to_node_idx = to_tok + partial_sum_arr[to_window]
        dist[from_node_idx][to_node_idx] = paths_per_windows[to_window][to_tok].score
        connection_type[from_node_idx][to_node_idx] = edge[0]
        
  for i in range(number_of_toks):
    dist[number_of_toks][i] = 0
  
  for window_idx in range(number_of_windows): #artificial start node
    for k in range(len(paths_per_windows[window_idx])):
      offset = partial_sum_arr[window_idx]
      dist[number_of_toks][offset + k] = paths_per_windows[window_idx][k].score

  V = number_of_toks + 1
  for i in range(V):
    for j in range(V):
        # No edge between node
        # i and j
        if (dist[i][j] == float('inf')):
            Next[i][j] = -1
        else:
            Next[i][j] = j
  
  #floyd warshal implementation of floyd warshall from https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/
  for k in range(V):
    for i in range(V):
        for j in range(V):
              
            # We cannot travel through
            # edge that doesn't exist
            if (dist[i][k] == float('inf') or dist[k][j] == float('inf')):
                continue
            if (dist[i][j] > dist[i][k] + dist[k][j]):
                dist[i][j] = dist[i][k] + dist[k][j]
                Next[i][j] = Next[i][k]

  from_node = len(dist) - 1
  dist[-1][-1] = float("inf")
  path = []
  path_lst = []
  paths_merged_lst = []
  break1 = False
  break2 = False
  while True:
    if path == []:
      to_node = np.argmin(dist[-1]) #last node is artificial start node 
      path = constructPath(from_node,to_node,Next)
      dist[-1][to_node] = float("inf")
      path_lst.append(path)
      paths_merged_lst.append(merge_paths_by_indexes(path_lst[-1][1:],partial_sum_arr,connection_type,paths_per_windows))
    else:
      if break1 == False:
        last_previous_node = path_lst[-1][-1]
        if np.min(dist[-1][last_previous_node+1:]) == float("inf"):
          #print("break")
          break1 = True
          #break
        else:
          to_node = last_previous_node + 1 + np.argmin(dist[-1][last_previous_node+1:]) #+1, argmin returns index from 0
          path2 = constructPath(from_node,to_node, Next)
          dist[-1][to_node] = float("inf")  

          if path2[1] > path_lst[-1][-1]:
            path_lst.append(path2)
            paths_merged_lst.append(merge_paths_by_indexes(path_lst[-1][1:],partial_sum_arr,connection_type,paths_per_windows))
            path = path2

      if break2 == False:
        first_previous_node = path_lst[0][1] 
        if np.min(dist[-1][0:first_previous_node]) == float("inf"):
            #print("break")
            break2 = True
            #break
        else:
          to_node =  np.argmin(dist[-1][0:first_previous_node]) #+1
          path2 = constructPath(from_node,to_node, Next)
          dist[-1][to_node] = float("inf")  
          path_lst.insert(0,path2)
          paths_merged_lst.insert(0,merge_paths_by_indexes(path_lst[0][1:],partial_sum_arr,connection_type,paths_per_windows))
          path = path2
      if break1 and break2:
          break

    if path_lst[-1][-1] >=  partial_sum_arr[-1]:
      break1 = True
    if path_lst[0][1] <=  0:
      break2 = True


  return paths_merged_lst,path_lst,partial_sum_arr
  
def get_max_y_for_interpolation(court_points_per_image,image_height):
  """
    Description:
      Remove paths that are in.horizontal base line zone
    Parameters:
      court_points_per_image       (list(tuple(int,int)))    : court points found in image     
      image_height                 (int)                     : image heigth in pixels
    Returns:
      max_y                        (float)                   : heigth of court in coord y
  """
  max_y = 0
  for points in court_points_per_image:
      if points is None:
        continue
        
      point0 = min(image_height,points[0][1])
      point10 = max(0,points[10][1])

      point17 = min(image_height,points[17][1])
      point18 = max(0,points[18][1])

      point3 = min(image_height,points[3][1])
      point13 = max(0,points[13][1])   
      
      max_y_tmp1 = abs(point0 - point10) 
      max_y_tmp2 = abs(point17 - point18)
      max_y_tmp3 = abs(point3 - point13)
      
      max_y = max(max_y,max_y_tmp1,max_y_tmp2,max_y_tmp3)
  return  max_y  
 
def run_TLDA(detections_in_frames, court_points_per_image, image_width, image_height, max_windows_forward = 8, MAX_DIST = 130):
  """
    Description:
      Find ball trajectory by TLDA algorithm from ball detections in frames and court information
    Parameters:
      detections_in_frames                    (list(list(tuple[int,int])))                        : list of lists of detections in frames, [#frame][#detection], format of detection is x,y
      court_points_per_image                  (list(tuple(int,int)))                              : court points found in image
      image_width                             (int)                                               : image width in pixels
      image_height                            (int)                                               : image heigth in pixels
      max_windows_forward = 8                 (int)                                               : base value for max windows forward to look for merging paths
      MAX_DIST = 130                          (int)                                               : max distance between detections in neighboring frames, this distance tested for 1280x720 video resolution

    Returns:
    trajectories_final[0].path_coords         (list(int,int,int))                                 : trajectory represented as frame,x,y       
  """
  
  max_y_diff_for_merge = get_max_y_for_interpolation(court_points_per_image,image_height)
  max_windows_forward_options = [max_windows_forward, max_windows_forward+2, max_windows_forward+4, max_windows_forward+8]


  windows, start_frame_of_windows               = make_windows(windows_size, window_shift, detections_in_frames)
  graph_edges_indexes_per_window                = make_connection_graph_per_window(windows)
  paths_per_windows                             = find_paths_in_windows(windows,graph_edges_indexes_per_window,start_frame_of_windows)
  paths_per_windows                             = merge_paths_inside_windows(windows,paths_per_windows, max_y_diff_for_merge)
  paths_per_windows                             = remove_paths_in_base_line_areas(paths_per_windows, court_points_per_image)
  paths_per_windows                             = reduce_number_of_paths_in_windows(paths_per_windows, upper_limit_on_paths_per_window)
  
  number_of_paths = 0
  for i in range(len(paths_per_windows)):
    number_of_paths = number_of_paths + len(paths_per_windows[i])
    
  for max_windows_forward in max_windows_forward_options:
    graph_paths_edges_indexes_per_window           = make_graph_for_paths_in_different_windows(windows,paths_per_windows,max_windows_forward, max_y_diff_for_merge)
    
    if VERSION == 'C_and_PyCUDA':
      if number_of_paths >= 800:
        cuda.init()
        device = cuda.Device(0) # enter your gpu id here
        ctx = device.make_context()
        trajectories_final,path_lst,partial_sum_arr   = find_shortest_lightest_trajectory_floyd_warshall_pycuda(paths_per_windows,graph_paths_edges_indexes_per_window) 
      else:
        trajectories_final,path_lst,partial_sum_arr   = find_shortest_lightest_trajectory_floyd_warshall_c(paths_per_windows,graph_paths_edges_indexes_per_window)
    elif VERSION == 'C':
      trajectories_final,path_lst,partial_sum_arr   = find_shortest_lightest_trajectory_floyd_warshall_c(paths_per_windows,graph_paths_edges_indexes_per_window)
    elif VERSION == 'PyCUDA':
      cuda.init()
      device = cuda.Device(0) # enter your gpu id here
      ctx = device.make_context()
      trajectories_final,path_lst,partial_sum_arr   = find_shortest_lightest_trajectory_floyd_warshall_pycuda(paths_per_windows,graph_paths_edges_indexes_per_window) 
    elif VERSION == 'Python':
      trajectories_final,path_lst,partial_sum_arr   = find_shortest_lightest_trajectory_floyd_warshall_python(paths_per_windows,graph_paths_edges_indexes_per_window)
    
    
    
    if len(trajectories_final) == 1:
      break
    else: #more than one trajectory
      for i in range(len(trajectories_final)-1,-1,-1):
        delete_trajectory = True
        for j in range(len(trajectories_final[i].path_coords)):
          current_frame_index = trajectories_final[i].path_coords[j][0]
          ball_position = (trajectories_final[i].path_coords[j][1],trajectories_final[i].path_coords[j][2])
          if is_inside_middle_lines_extended_area(ball_position,court_points_per_image,current_frame_index, image_width, image_height) == 'in':
            delete_trajectory = False
            break
        if delete_trajectory == True:
          trajectories_final.pop(i)
          
      sum_of_lens = 0
      for i in range(len(trajectories_final)):
        sum_of_lens = sum_of_lens + len(trajectories_final[i].path_coords)
        
      for i in range(len(trajectories_final)):
        if len(trajectories_final[i].path_coords) >= (sum_of_lens * 0.75): #one trajectory long enough, over 0.75% of sum of all lengths, rest is noise probably
          return trajectories_final[i].path_coords
    
  
  if VERSION == 'PyCUDA' or VERSION == 'C_and_PyCUDA':
    ctx.pop()
  #delete trajectory that doesnt have detection middle lines extendes area
  for i in range(len(trajectories_final)-1,-1,-1):
    delete_trajectory = True
    for j in range(len(trajectories_final[i].path_coords)):
      current_frame_index = trajectories_final[i].path_coords[j][0]
      ball_position = (trajectories_final[i].path_coords[j][1],trajectories_final[i].path_coords[j][2])
      if is_inside_middle_lines_extended_area(ball_position,court_points_per_image,current_frame_index, image_width, image_height) == 'in':
        delete_trajectory = False
        break
    if delete_trajectory == True:
      trajectories_final.pop(i)
  
  if len(trajectories_final) > 1:#get longest one
    length_of_paths = []
    for i in range(len(trajectories_final)):
      length_of_paths.append(len(trajectories_final[i].path_coords))
    return trajectories_final[length_of_paths.index(max(length_of_paths))].path_coords
  elif len(trajectories_final) == 0:
    return None
  return trajectories_final[0].path_coords
  
  
  

  
  
   
  
  
  
