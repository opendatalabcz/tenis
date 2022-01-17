import os
import cv2
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from joblib import dump, load
from functools import cmp_to_key
import sys

sys.path.append('.')
from utils import get_intersection_of_two_lines, dot, length, vector, unit, distance, scale, add, get_intersection_of_two_lines_segments    


from multiprocessing import Process, Pipe, cpu_count, current_process, Manager,Pool
from time import process_time, sleep, time
import functools


#####CONSTS########
ANGLE_TOLERATION_TO_MERGE_LINES = 3
###################

class DisjSet:
  """Class for data structure disjoint set.

    Class for data structure disjoint set.
    implementation from https://www.geeksforgeeks.org/disjoint-set-data-structures/  

    Attributes:
      n        (int)   : number of members of structure
  """
  def __init__(self, n):
      # Constructor to create and
      # initialize sets of n items
      self.rank = [1] * n
      self.parent = [i for i in range(n)]


  # Finds set of given item x
  def find(self, x):
        
      # Finds the representative of the set
      # that x is an element of
      if (self.parent[x] != x):
            
          # if x is not the parent of itself
          # Then x is not the representative of
          # its set,
          self.parent[x] = self.find(self.parent[x])
            
          # so we recursively call Find on its parent
          # and move i's node directly under the
          # representative of this set

      return self.parent[x]


  # Do union of two sets represented
  # by x and y.
  def Union(self, x, y):
        
      # Find current sets of x and y
      xset = self.find(x)
      yset = self.find(y)

      # If they are already in same set
      if xset == yset:
          return

      # Put smaller ranked item under
      # bigger ranked item if ranks are
      # different
      if self.rank[xset] < self.rank[yset]:
          self.parent[xset] = yset

      elif self.rank[xset] > self.rank[yset]:
          self.parent[yset] = xset

      # If ranks are same, then move y under
      # x (doesn't matter which one goes where)
      # and increment rank of x's tree
      else:
          self.parent[yset] = xset
          self.rank[xset] = self.rank[xset] + 1
  


def calculate_hsv_histogram(image_hsv):
  """
    Description:
      Calculate h,s,v histogram for image in HSV color format.
    Parameters:
      image_hsv                     (numpy.array(width, height, depth))                                    : HSV image
    Returns:
      histograms_for_images      ((list(numpy.ndarray),list(numpy.ndarray),list(numpy.ndarray)))        : histograms of image
  """

  h, s, v = image_hsv[:,:,0], image_hsv[:,:,1], image_hsv[:,:,2]
  hist_h = cv2.calcHist([h],[0],None,[180],[0,179])
  hist_s = cv2.calcHist([s],[0],None,[256],[0,255])
  hist_v = cv2.calcHist([v],[0],None,[256],[0,255])
  return hist_h,hist_s,hist_v


def get_mask_of_court_lines(image_hsv, hist_v, hist_s, bins_to_climb_value, bins_to_climb_saturation):
  """
    Description:
      Calculate mask of court lines by thresholding value and saturation in HSV format.
      Thresholds obtained by method from Rea, N., Dahyot, R., Kokaram, A.: Classification and Representation of Semantic Content in Broadcast Tennis Videos
    Parameters:
      image_hsv                   (numpy.array(width,height, depth))         : HSV image
      hist_v                      (numpy.ndarray(256,1))                     : value histogram of HSV image
      hist_s                      (numpy.ndarray(256,1))                     : saturation histogram of HSV image
      bins_to_climb_value        (int)                                       : bins to climb in value histogram
      bins_to_climb_saturation   (int)                                       : bins to climb in saturation histogram
    Returns:
      final_mask                  (numpy.array(width,height, 1))      : mask of court lines, not perfect, needs another processing
  """

  index_v = int(np.argmax(hist_v, axis=0)[0])
  index_s = int(np.argmax(hist_s, axis=0)[0])

  index_right_v = index_v + 1
  index_left_v =  index_v - 1
  index_right_s = index_s + 1
  index_left_s =  index_s - 1
  for i in range(bins_to_climb_value):
    if hist_v[index_right_v] > hist_v[index_left_v]:
      index_right_v = index_right_v + 1
    else:
      index_left_v = index_left_v - 1
  for i in range(bins_to_climb_saturation):
    if hist_s[index_right_s] > hist_s[index_left_s]:
      index_right_s = index_right_s + 1
    else:
      index_left_s = index_left_s - 1

  frame_threshold_value = cv2.inRange(image_hsv, (0, 0, index_right_v), (179, 255, 255))
  frame_threshold_saturation = cv2.inRange(image_hsv, (0, 0, 0), (179, index_left_s, 255))
  final_mask = cv2.bitwise_and(frame_threshold_saturation,frame_threshold_value)

  return final_mask

def get_lines_from_mask(mask, minLineLength):
  """
    Description:
      Find lines in mask by probabilistic hough line transforamtion.
    Parameters:
      mask                    (numpy.array(width,height, 1))            : mask of court lines
      minLineLength           (int)                                     : minimum length of line detected in mask
    Returns:
      lines                   (numpy.array(lines_cnt,4))                : lines for mask, lines in format of two end points (x1,y1),(x2,y2)
  """
  #params not tuned
  #minLineLength = 100
  maxLineGap = 20
  pixel_granularity = 1
  angle_granularity = np.pi/180
  votes_threshold = minLineLength
  lines = cv2.HoughLinesP(mask,pixel_granularity,angle_granularity,votes_threshold,minLineLength = minLineLength,maxLineGap = maxLineGap)

  if lines is None:
    return None
  lines = lines[:,0,:]
  return lines




  

def is_in_distance_with_toleration_L2(pnt, start, end, toleration_L2):
  """
    Description:
      Decide if point is in tolerated L2 distance from line segment.
      Distance calculation from http://paulbourke.net/geometry/pointlineplane/
    Parameters:
      pnt                   (tuple(int,int)): point
      start                 (tuple(int,int)): start point of line
      end                   (tuple(int,int)): end point of line
      toleration_L2         (int)           : toleration in L2 distance
    Returns:
                            (bool)          : true if distance in toleration, false otherwise
  """
  line_vec = vector(start, end)
  pnt_vec = vector(start, pnt)
  line_len = length(line_vec)
  line_unitvec = unit(line_vec)
  pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
  t = dot(line_unitvec, pnt_vec_scaled)    
  if t < 0.0:
      t = 0.0
  elif t > 1.0:
      t = 1.0
  nearest = scale(line_vec, t)
  dist = distance(nearest, pnt_vec)
  if dist <= toleration_L2:
    return True
  else:
    return False




def are_lines_touching(line1,line2, toleration = 10):
  """
    Description:
      Are lines touching with toleration
    Parameters:
      line1                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
      line2                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
      toleration              (int)                   : toleration in distance
    Returns:
                            (bool)                    : true if lines are touching within toleration
  """
  
  is_intersection = False
  px,py = get_intersection_of_two_lines((line1[0],line1[1]),(line1[2],line1[3]),(line2[0],line2[1]),(line2[2],line2[3]))
  if px != -1 and py != -1:
    is_intersection = True
  if not is_intersection:
    return False
  intersection_point = (px,py)
  
  dist1 = is_in_distance_with_toleration_L2(intersection_point, (line1[0],line1[1]),(line1[2],line1[3]), toleration)
  dist2 = is_in_distance_with_toleration_L2(intersection_point, (line2[0],line2[1]),(line2[2],line2[3]), toleration)

  if dist1 and dist2:
    return True
  else:
    return False

    
def is_in_distance_with_toleration(point1, point2, toleration_x, toleration_y):
  """
    Description:
      Are points within toleration distance
    Parameters:
      point1                   (tuple(int,int,int,int)): point represented by (x,y)
      point2                   (tuple(int,int,int,int)): point represented by (x,y)
      toleration_x             (int)                   : toleration in distance for x
      toleration_y             (int)                   : toleration in distance for y
    Returns:
                               (bool)                  : true if points are in tolerated distance
  """
  dist_x = math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]))
  dist_y = math.sqrt((point1[1] - point2[1]) * (point1[1] - point2[1]) )
  if dist_x <= toleration_x and dist_y <= toleration_y:
    return True
  else:
    return False
def dist(point1, point2):
  return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]) )
def dist_x(point1, point2):
  return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0])  )
def dist_y(point1, point2):
  return math.sqrt( (point1[1] - point2[1]) * (point1[1] - point2[1]) )


def sort_by_x_comp(line1,line2):
  """
    Description:
      Compare lines by x coord
    Parameters:
      line1                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
      line2                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
    Returns:
                            (bool)                    : true if first line has higher x coord
  """
  return min(line1[0],line1[2]) - min(line2[0],line2[2])
def sort_by_y_comp(line1,line2):
  """
    Description:
      Compare lines by x coord
    Parameters:
      line1                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
      line2                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
    Returns:
                            (bool)                    : true if first line has higher y coord
  """
  return min(line1[1],line1[3]) - min(line2[1],line2[3])

def should_merge_lines(line1, line2):
  """
    Description:
      Decide if merge two lines if they have similiar angle and lines are touching with toleration
    Parameters:
      line1                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
      line2                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
    Returns:
                            (bool)                    : true if first line has higher y coord
  """
  x1,y1,x2,y2 = line1
  x3,y3,x4,y4 = line2
  vector_1 = [x2-x1,y2-y1]
  vector_2 = [x4-x3,y4-y3]
  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)
  if dot_product > 1:
    dot_product = 1
  elif dot_product < -1:
    dot_product = -1
  angle = np.arccos(dot_product) *  180 / np.pi
  

  middle_point1_x = (x1 + x2) / 2
  middle_point1_y = (y1 + y2) / 2
  middle_point1 = (middle_point1_x,middle_point1_y)
  middle_point2_x = (x3 + x4) / 2
  middle_point2_y = (y3 + y4) / 2
  middle_point2 = (middle_point2_x,middle_point2_y)

  toleration_x_1 = max(5,abs( (x1 - x2) / 2 ))# max if diffrence between end points is very small, eg. 0
  toleration_y_1 = max(5,abs( (y1 - y2) / 2))
  toleration_x_2 = max(5,abs( (x3 - x4) / 2))
  toleration_y_2 = max(5,abs( (y3 - y4) / 2))

  dist1 = is_in_distance_with_toleration(middle_point1,middle_point2,toleration_x_1,toleration_y_1)
  
  dist2 = is_in_distance_with_toleration(middle_point1,(x3,y3),toleration_x_1,toleration_y_1)
  dist3 = is_in_distance_with_toleration(middle_point1,(x4,y4),toleration_x_1,toleration_y_1)

  dist4 = is_in_distance_with_toleration(middle_point2,middle_point1,toleration_x_2,toleration_y_2)
  
  dist5 = is_in_distance_with_toleration(middle_point2,(x1,y1),toleration_x_2,toleration_y_2)
  dist6 = is_in_distance_with_toleration(middle_point2,(x2,y2),toleration_x_2,toleration_y_2)


  if angle <= ANGLE_TOLERATION_TO_MERGE_LINES and (dist1 or dist2 or dist3 or dist4 or dist5 or dist6):
    return True
  else:
    return False

def merge_lines(equivalence_classes_indexes,lines):
  """
    Description:
      Merge lines into one
    Parameters:
      equivalence_classes_indexes                   (dict(int,list(int)))   : dictionary with list with indexes of lines that should be merged to one
      lines                                         (list(int,int,int,int)) : list of lines represented by two points (x,y),(x,y)
    Returns:
      merged_lines                                  (list(int,int,int,int)) : list of merged lines
  """

  merged_lines = []
  for key,equivalence_class_indexes in equivalence_classes_indexes.items():
    vertical_cnt = 0
    horizontal_cnt = 0
    points = []
    for index in equivalence_class_indexes:
      x1,y1,x2,y2 = lines[index]
      points.append((x1,y1)) 
      points.append((x2,y2))
      if abs(x1-x2) < abs(y1-y2):
        vertical_cnt = vertical_cnt + 1
      elif abs(x1-x2) > abs(y1-y2):
        horizontal_cnt = horizontal_cnt + 1


    if horizontal_cnt > vertical_cnt:#merge by x coord
      biggest_x = points[0][0]
      smallest_x = points[0][0]
      y_coords_for_biggest_x = [points[0][1]]
      y_coords_for_smallest_x = [points[0][1]]

      for i in range(1,len(points)):
        if points[i][0] > biggest_x:
          biggest_x = points[i][0]
          y_coords_for_biggest_x = [points[i][1]]
        elif points[i][0] == biggest_x: #same biggest x, append y coord, afterwards would be averaged
          y_coords_for_biggest_x.append( points[i][1])

        if points[i][0] < smallest_x:
          smallest_x = points[i][0]
          y_coords_for_smallest_x = [points[i][1]]
        elif points[i][0] == smallest_x: #same smallest x, append y coord, afterwards would be averaged
          y_coords_for_smallest_x.append( points[i][1] ) 

      y_for_smallest_x = int(sum(y_coords_for_smallest_x) / len(y_coords_for_smallest_x))
      y_for_biggest_x = int(sum(y_coords_for_biggest_x) / len(y_coords_for_biggest_x))
      merged_lines.append([smallest_x,y_for_smallest_x,biggest_x,y_for_biggest_x])
    elif horizontal_cnt < vertical_cnt: #merge by y-coord
      biggest_y = points[0][1]
      smallest_y = points[0][1]
      x_coord_for_biggest_y = [points[0][0]]
      x_coord_for_smallest_y = [points[0][0]]
      
      for i in range(1,len(points)):
        if points[i][1] > biggest_y:
          biggest_y = points[i][1]
          x_coord_for_biggest_y = [points[i][0]]
        elif points[i][1] == biggest_y: #same biggest y, append x coord, afterwards would be averaged
          x_coord_for_biggest_y.append(points[i][0])
        if points[i][1] < smallest_y:
          smallest_y = points[i][1]
          x_coord_for_smallest_y = [points[i][0]]
        elif points[i][1] == smallest_y: #same biggest y, append x coord, afterwards would be averaged
          x_coord_for_smallest_y.append(points[i][0])

      x_for_smallest_y = int(sum(x_coord_for_smallest_y) / len(x_coord_for_smallest_y))
      x_for_biggest_y = int(sum(x_coord_for_biggest_y) / len(x_coord_for_biggest_y))
      merged_lines.append([x_for_smallest_y,smallest_y,x_for_biggest_y,biggest_y])

  return merged_lines




def should_merge_non_touching_horizontal_lines(line1,line2):
  """
    Description:
      Decide if merge non touching horizotntal lines. For example court line could be halfed by occlusion from player
      Lengthen the line by its vector a decide if then the lines are touching
    Parameters:
      line1                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
      line2                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
    Returns:
                              (bool)                  : True if lines should be merged
  """
  x1,y1,x2,y2 = line1
  x3,y3,x4,y4 = line2
  vector_1 = [x2-x1,y2-y1]
  vector_2 = [x4-x3,y4-y3]
  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)
  
  if dot_product > 1:
    dot_product = 1
  elif dot_product < -1:
    dot_product = -1
  angle = np.arccos(dot_product) *  180 / np.pi
  should_merge_by_angle = angle <= ANGLE_TOLERATION_TO_MERGE_LINES

  distances_in_x = [abs(line1[0] - line2[0]),abs(line1[0] - line2[2]),abs(line1[2] - line2[0]),abs(line1[2] - line2[2])]
  case = distances_in_x.index(min(distances_in_x))
  merged_line = []
  should_merge = False
  #######################
  tolerance = 2 #tolerance for lengthening the line by distance between two lines
  tolerance_py = 10 #low threshold
  ########################
  shift_in_y_to_shift_in_x = max(abs(line1[1] - line1[3]) / abs(line1[0] - line1[2]), abs(line2[1] - line2[3]) / abs(line2[0] - line2[2])) #how big is shift in x for one pixel shift in y
  #case by closest points from two lines
  if case == 0: 
    dist_in_x = abs(line1[0] - line2[0])
    tolerance_on_y_dist = shift_in_y_to_shift_in_x * dist_in_x * tolerance
    if abs(line1[1] - line2[1]) < tolerance_on_y_dist or abs(line1[1] - line2[1]) < tolerance_py:
      should_merge = True
  if case == 1: 
    dist_in_x = abs(line1[0] - line2[2])
    tolerance_on_y_dist = shift_in_y_to_shift_in_x * dist_in_x * tolerance
    if abs(line1[1] - line2[3]) < tolerance_on_y_dist or abs(line1[1] - line2[3]) < tolerance_py:
      should_merge = True
  if case == 2: 
    dist_in_x = abs(line1[2] - line2[0])
    tolerance_on_y_dist = shift_in_y_to_shift_in_x * dist_in_x * tolerance
    if abs(line1[3] - line2[1]) < tolerance_on_y_dist or abs(line1[3] - line2[1]) < tolerance_py:
      should_merge = True
  if case == 3: 
    dist_in_x = abs(line1[2] - line2[2])
    tolerance_on_y_dist = shift_in_y_to_shift_in_x * dist_in_x * tolerance
    if abs(line1[3] - line2[3]) < tolerance_on_y_dist or abs(line1[3] - line2[3]) < tolerance_py:
      should_merge = True
  return (should_merge and should_merge_by_angle)

def should_merge_non_touching_vertical_lines(line1,line2):
  """
    Description:
      Decide if merge non touching vertical lines. For example court line could be halfed by occlusion from player
      Lengthen the line by its vector a decide if then the lines are touching
    Parameters:
      line1                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
      line2                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
    Returns:
                              (bool)                  : True if lines should be merged
  """
  x1,y1,x2,y2 = line1
  x3,y3,x4,y4 = line2
  vector_1 = [x2-x1,y2-y1]
  vector_2 = [x4-x3,y4-y3]
  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)
  
  if dot_product > 1:
    dot_product = 1
  elif dot_product < -1:
    dot_product = -1
  angle = np.arccos(dot_product) *  180 / np.pi
  should_merge_by_angle = angle <= ANGLE_TOLERATION_TO_MERGE_LINES

  distances_in_y = [abs(line1[1] - line2[1]),abs(line1[1] - line2[3]),abs(line1[3] - line2[1]),abs(line1[3] - line2[3])] #merge vertical find two points closest in y cord
  case = distances_in_y.index(min(distances_in_y))
  merged_line = []
  should_merge = False
  ###################################
  tolerance = 2     #tolerance for lengthening the line
  tolerance_px = 10 #low threshold
  ##################################
  shift_in_x_to_shift_in_y = max(abs(line1[0] - line1[2]) / abs(line1[1] - line1[3]), abs(line2[0] - line2[2]) / abs(line2[1] - line2[3])) #vzdalenost "x" lomeno vzdalenost "y", tedy kolik pixelu "x" na jeden posun v y
  if case == 0: 
    dist_in_y = abs(line1[1] - line2[1])
    tolerance_on_x_dist = shift_in_x_to_shift_in_y * dist_in_y * tolerance
    if abs(line1[0] - line2[0]) < tolerance_on_x_dist or abs(line1[0] - line2[0]) < tolerance_px:
      should_merge = True
  if case == 1: 
    dist_in_y = abs(line1[1] - line2[3])
    tolerance_on_x_dist = shift_in_x_to_shift_in_y * dist_in_y * tolerance
    if abs(line1[0] - line2[2]) < tolerance_on_x_dist or abs(line1[0] - line2[2]) < tolerance_px:
      should_merge = True
  if case == 2: 
    dist_in_y = abs(line1[3] - line2[1])
    tolerance_on_x_dist = shift_in_x_to_shift_in_y * dist_in_y * tolerance
    if abs(line1[2] - line2[0]) < tolerance_on_x_dist or abs(line1[2] - line2[0]) < tolerance_px:
      should_merge = True
  if case == 3: 
    dist_in_y = abs(line1[3] - line2[3])
    tolerance_on_x_dist = shift_in_x_to_shift_in_y * dist_in_y * tolerance
    if abs(line1[2] - line2[2]) < tolerance_on_x_dist or abs(line1[2] - line2[2]) < tolerance_px:
      should_merge = True
  return (should_merge and should_merge_by_angle)



def count_vertical_and_horizontal(indexes, lines):
  """
    Description:
      Count horizontal and vertical lines on specified indexes
    Parameters:
      equivalence_classes_indexes                   (dict(int,list(int)))   : dictionary with list with indexes of lines that should be merged to one
      lines                                         (list(int,int,int,int)) : list of lines represented by two points (x,y),(x,y)
    Returns:
      (vertical_cnt,horizontal_cnt)                 (tuple(int,int))        : number of vertical lines, number of horizontal lines
  """
  vertical_cnt = 0
  horizontal_cnt = 0
  for index in indexes:
    x1,y1,x2,y2 = lines[index]
    if abs(x1-x2) < abs(y1-y2):
      vertical_cnt = vertical_cnt + 1
    elif abs(x1-x2) > abs(y1-y2):
      horizontal_cnt = horizontal_cnt + 1
  return (vertical_cnt,horizontal_cnt)


def build_equivalence_classes_of_touching_lines_with_similiar_angle(lines):
  """
    Description:
      Put in one class lines that should be merged by similiar angle and closeness. For classes is used disjoint set.
      Merging lines that are part of one court line
    Parameters:
      lines                                         (list(int,int,int,int)) : list of lines represented by two points (x,y),(x,y)
    Returns:
      (equivalence_classes)                         (list(list(int))        : list of classes represented by list of indexes to parameter "lines"
  """
  number_of_lines = len(lines)
  obj = DisjSet(number_of_lines)
  for i in range(number_of_lines):
    for j in range(i,number_of_lines):
      if should_merge_lines(lines[i],lines[j]):
        obj.Union(i,j)

  equivalence_classes = {}
  for i in range(number_of_lines):
    parrent = obj.find(i)
    if parrent in equivalence_classes.keys():
      equivalence_classes[parrent].append(i)
    else:
      equivalence_classes[parrent] = [i]
  return equivalence_classes

  
def build_equivalence_classes_of_touching_lines(lines):
  """
    Description:
      Put in one class lines that should be merged by closeness. similiar angle is not needed  For classes is used disjoint set.
      For merging lines that are touching, but similiar angle is not needed, so horizontal and vertical line can be merged.
    Parameters:
      lines                                         (list(int,int,int,int)) : list of lines represented by two points (x,y),(x,y)
    Returns:
      (equivalence_classes)                         (list(list(int))        : list of classes represented by list of indexes to parameter "lines"
  """
  number_of_lines = len(lines)
  obj = DisjSet(number_of_lines)
  for i in range(number_of_lines):
    for j in range(i,number_of_lines):
      if are_lines_touching(lines[i],lines[j]):
        obj.Union(i,j)

  equivalence_classes = {}
  for i in range(number_of_lines):
    parrent = obj.find(i)
    if parrent in equivalence_classes.keys():
      equivalence_classes[parrent].append(i)
    else:
      equivalence_classes[parrent] = [i]
  return equivalence_classes


def build_equivalence_classes_of_non_touching_horizontal_lines_that_should_be_merged(lines):
  """
    Description:
      Put in one class lines that should be merged even if their are not touching. For classes is used disjoint set.
      For merging horizontal lines that are not touching, for example part of line is occluded by player
    Parameters:
      lines                                         (list(int,int,int,int)) : list of lines represented by two points (x,y),(x,y)
    Returns:
      (equivalence_classes)                         (list(list(int))        : list of classes represented by list of indexes to parameter "lines"
  """
  number_of_lines = len(lines)
  obj = DisjSet(number_of_lines)
  for i in range(number_of_lines):
    for j in range(i,number_of_lines):
      if should_merge_non_touching_horizontal_lines(lines[i],lines[j]):
        obj.Union(i,j)

  equivalence_classes = {}
  for i in range(number_of_lines):
    parrent = obj.find(i)
    if parrent in equivalence_classes.keys():
      equivalence_classes[parrent].append(i)
    else:
      equivalence_classes[parrent] = [i]
  return equivalence_classes

def build_equivalence_classes_of_non_touching_vertical_lines_that_should_be_merged(lines):
  """
    Description:
      Put in one class lines that should be merged even if their are not touching. For classes is used disjoint set.
      For merging vertical lines that are not touching, for example part of line is occluded by player
    Parameters:
      lines                                         (list(int,int,int,int)) : list of lines represented by two points (x,y),(x,y)
    Returns:
      (equivalence_classes)                         (list(list(int))        : list of classes represented by list of indexes to parameter "lines"
  """
  number_of_lines = len(lines)
  obj = DisjSet(number_of_lines)
  for i in range(number_of_lines):
    for j in range(i,number_of_lines):
      if should_merge_non_touching_vertical_lines(lines[i],lines[j]):
        obj.Union(i,j)

  equivalence_classes = {}
  for i in range(number_of_lines):
    parrent = obj.find(i)
    if parrent in equivalence_classes.keys():
      equivalence_classes[parrent].append(i)
    else:
      equivalence_classes[parrent] = [i]
  return equivalence_classes



def separate_horizontal_and_vertical_lines(indexes, lines):
  """
    Description:
      Separate horizontal and vertical lines.
    Parameters:
      indexes                                       (list(int))               : indexes of lines, that should be examined
      lines                                         (list(int,int,int,int))   : list of lines represented by two points (x,y),(x,y)
    Returns:
      (vertical_lines,horizontal_lines)             (tuple(int,int))          : separated horizontal and vertical lines
  """
  
  vertical_lines = []
  horizontal_lines = []
  for index in indexes:
    x1,y1,x2,y2 = lines[index]
    if abs(x1-x2) < abs(y1-y2):
      vertical_lines.append(lines[index])
    elif abs(x1-x2) > abs(y1-y2):
      horizontal_lines.append(lines[index])
  return (vertical_lines,horizontal_lines)


def find_horizontal_and_vertical_lines(lines):
  """
    Description:
      Find horizontal and vertical lines of court
    Parameters:
      lines                                         (list(int,int,int,int))   : list of lines represented by two points (x,y),(x,y)
    Returns:
      (vertical_lines,horizontal_lines)             (tuple(int,int))          : separated horizontal and vertical lines
  """
  # equivalence classes of lines touching and has almost same angle
  equivalence_classes = build_equivalence_classes_of_touching_lines_with_similiar_angle(lines)
  #merge lines   
  merged_lines = merge_lines(equivalence_classes,lines)
  # equivalence classes of lines touching and dont need almost same angle
  equivalence_classes = build_equivalence_classes_of_touching_lines(merged_lines)

  court_line_candidates = []
  for key, item in equivalence_classes.items():
  
    vertical_lines,horizontal_lines = separate_horizontal_and_vertical_lines(item, merged_lines)

    equivalence_classes_indexes = build_equivalence_classes_of_non_touching_horizontal_lines_that_should_be_merged(horizontal_lines)
    horizontal_lines = merge_lines(equivalence_classes_indexes,horizontal_lines)

    equivalence_classes_indexes = build_equivalence_classes_of_non_touching_vertical_lines_that_should_be_merged(vertical_lines)
    vertical_lines = merge_lines(equivalence_classes_indexes,vertical_lines)

    vertical_cnt = len(vertical_lines)
    horizontal_cnt = len(horizontal_lines)
    
    if (4 <= vertical_cnt <=5) and (5 <= horizontal_cnt <=6): #net represented by one or two lines, depends on straightness of the net

      vertical_lines = sorted(vertical_lines, key=cmp_to_key(sort_by_x_comp), reverse=False) #vertical from left
      horizontal_lines = sorted(horizontal_lines, key=cmp_to_key(sort_by_y_comp),reverse=True) #horizontal from bottom
      
      are_lines_intersected = False
      for i in range(len(horizontal_lines)):
        for j in range(i+1,len(horizontal_lines)):
          if are_lines_touching(horizontal_lines[i],horizontal_lines[j]):
            are_lines_intersected = True

      for i in range(len(vertical_lines)):
        for j in range(i+1,len(vertical_lines)):
          if are_lines_touching(vertical_lines[i],vertical_lines[j]):
            are_lines_intersected = True
      # horizontal lines can intersect each other, same for vertical    
      if are_lines_intersected == True:
        continue


      if horizontal_cnt == 5:#check for case, when top base line not detected and detected net by two line
        y_horiz = min(horizontal_lines[4][1],horizontal_lines[4][3])#fifth line should be baseline
        y_ver1 = min(vertical_lines[1][1],vertical_lines[1][3])
        y_ver2 = min(vertical_lines[-2][1],vertical_lines[-2][3])
        y_ver3 = min(vertical_lines[0][1],vertical_lines[0][3])
        y_ver4 = min(vertical_lines[-1][1],vertical_lines[-1][3])
        
        #if coord y of top baseline is +/- same as coord y of higher point any vertical line
        if not(abs(y_horiz - y_ver1) < 10 or abs(y_horiz - y_ver2) < 10 or abs(y_horiz - y_ver3) < 10 or abs(y_horiz - y_ver4) < 10): 
          continue
          
      elif horizontal_cnt == 6:
        x_min_net1 = min(horizontal_lines[2][0],horizontal_lines[2][2])
        x_min_net2 = min(horizontal_lines[3][0],horizontal_lines[3][2])
        #net by two lines, sort so on position 3 is line representing left part of the net
        if x_min_net1 > x_min_net2: 
          tmp = horizontal_lines[2]
          horizontal_lines[2] = horizontal_lines[3]
          horizontal_lines[3] = tmp

        
        y_dist_nets = get_y_dist_between_two_lines(horizontal_lines[2], horizontal_lines[3])
        y_dist_bottom_baseline_and_bottom_serveline = get_y_dist_between_two_lines(horizontal_lines[0],horizontal_lines[1])
        y_dist_top_baseline_and_top_serveline = get_y_dist_between_two_lines(horizontal_lines[4],horizontal_lines[5])
        #distance in y coord between two lines representing nets should be lower than distance in y coord between top base line and top serve line 
        #and lower than  than distance in y coord between bottom base line and bottom serve line
        if y_dist_nets >  y_dist_bottom_baseline_and_bottom_serveline or y_dist_nets > y_dist_top_baseline_and_top_serveline:
          continue
          
      if vertical_cnt  == 4:
        is_correct = True
        for i in range(len(horizontal_lines)):
          x1,y1,x2,y2 = horizontal_lines[i]
          x_middle = (x1+x2)/2
          for j in range(len(vertical_lines)):
            x3,y3,x4,y4 = vertical_lines[j]
            dist_end_point_to_end_point = min(abs(x1 - x3),abs(x2 - x3),abs(x1 - x4),abs(x2 - x4))
            dist_end_point_to_middle_point = min(abs(x_middle - x3),abs(x_middle - x4))
            #if all verticals in x are closer to end points than middle of horizontal, so in the 4 vertical lines is not middle vertical line
            if dist_end_point_to_end_point > dist_end_point_to_middle_point:
              is_correct = False
        if is_correct == False:
          continue  
        
      court_line_candidates.append((vertical_lines,horizontal_lines))

  if len(court_line_candidates) == 0:
    return None, None
    #choose biggest one court candidate if more detected
  if len(court_line_candidates) > 1:    
    index_of_candidate = 0
    best_len_of_lines = 0
    for i in range(len(court_line_candidates)):
      court_candidate = court_line_candidates[i]
      len_of_lines = 0
      for line in court_line_candidates[0][0]:
        len_of_lines = len_of_lines + dist((line[0],line[1]),(line[2],line[3]))
      for line in court_line_candidates[0][1]:
        len_of_lines = len_of_lines + dist((line[0],line[1]),(line[2],line[3]))
      if len_of_lines > best_len_of_lines:
        index_of_candidate = i
        best_len_of_lines = len_of_lines

  vertical_lines = court_line_candidates[0][0]
  horizontal_lines = court_line_candidates[0][1]
  return horizontal_lines, vertical_lines


def get_y_dist_between_two_lines(line1,line2):
  """
    Description:
      Get y distance between two lines
    Parameters:
      line1                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
      line2                   (tuple(int,int,int,int)): line represented by two points (x,y),(x,y)
    Returns:
      merged_lines                                  (list(int,int,int,int)) : list of merged lines
  """
  x1,y1,x2,y2 = line1
  x3,y3,x4,y4 = line2
  return min(abs(y1-y3),abs(y1-y4),abs(y2-y3),abs(y2-y4))


def classify_court_lines(horizontal_lines, vertical_lines):
  """
    Description:
      Classify lines to specific court lines
    Parameters:
      horizontal_lines                                (list(int,int,int,int)) : list of lines represented by two points (x,y),(x,y)
      vertical_lines                                  (list(int,int,int,int)) : list of lines represented by two points (x,y),(x,y)  
    Returns:
      court_lines                                     (dictionary(string:line)) : list of merged lines, line represented as (int,int,int,int) aka (x,y),(x,y)
  """
  court_lines = {
    'base_bottom':[],
    'service_bottom':[],
    'base_top':[],
    'service_top':[],
    'vertical_double_left':[],
    'vertical_single_left':[],
    'vertical_center' :[],
    'vertical_double_right':[],
    'vertical_single_right':[],
    'net1':[],
    'net2':[],
    }
  if len(horizontal_lines) == 6: 
    court_lines['base_bottom'] =    horizontal_lines[0]
    court_lines['service_bottom'] = horizontal_lines[1]
    court_lines['net1'] = horizontal_lines[2]
    court_lines['net2'] = horizontal_lines[3]
    court_lines['service_top'] =    horizontal_lines[4]
    court_lines['base_top'] =       horizontal_lines[5]

  elif len(horizontal_lines) == 5:
    court_lines['base_bottom'] =    horizontal_lines[0]
    court_lines['service_bottom'] = horizontal_lines[1]
    court_lines['net1'] = horizontal_lines[2]
    court_lines['service_top'] =    horizontal_lines[3]
    court_lines['base_top'] =       horizontal_lines[4]

  if len(vertical_lines) == 5:              
    court_lines['vertical_double_left'] =   vertical_lines[0]
    court_lines['vertical_single_left'] =   vertical_lines[1]
    court_lines['vertical_center'] =        vertical_lines[2]
    court_lines['vertical_single_right'] =  vertical_lines[3]
    court_lines['vertical_double_right'] =  vertical_lines[4]
  elif len(vertical_lines) == 4:
    court_lines['vertical_double_left'] =   vertical_lines[0]
    court_lines['vertical_single_left'] =   vertical_lines[1]
    court_lines['vertical_center'] =        []
    court_lines['vertical_single_right'] =  vertical_lines[2]
    court_lines['vertical_double_right'] =  vertical_lines[3]
  return court_lines


def find_court_corner_points(court_lines_classificated):
  """
    Description:
      Find corner points and other specific points of tennis court from lines
    Parameters:
      court_lines_classificated                       (dictionary(string:line)) : list of merged lines, line represented as (int,int,int,int) aka (x,y),(x,y)
    Returns:
      corner_points                                   (list(tuple(int,int)))    : list of intersection points between lines representing cornes and other significant court points
      corner_points_net                               (list(tuple(int,int)))    : list of intersection points between net lines and vertical lines and net lines themselves
  """

  corner_points = []
  corner_points_net = []
  if court_lines_classificated['vertical_center'] == []:
    number_of_corners = 12
    lines_to_intersect_for_corners = [('base_bottom','vertical_double_right'),('base_bottom','vertical_single_right'),('base_bottom','vertical_single_left'),\
                                    ('base_bottom','vertical_double_left'),
                                    ('service_bottom','vertical_single_right'),('service_bottom','vertical_single_left'),\
                                    ('service_top','vertical_single_right'),('service_top','vertical_single_left'),\
                                    ('base_top','vertical_double_right'),('base_top','vertical_single_right'),('base_top','vertical_single_left'),('base_top','vertical_double_left')]
  else:
    number_of_corners = 14
    lines_to_intersect_for_corners = [('base_bottom','vertical_double_right'),('base_bottom','vertical_single_right'),('base_bottom','vertical_single_left'),\
                                    ('base_bottom','vertical_double_left'),
                                    ('service_bottom','vertical_single_right'),('service_bottom','vertical_center'),('service_bottom','vertical_single_left'),\
                                    ('service_top','vertical_single_right'),('service_top','vertical_center'),('service_top','vertical_single_left'),\
                                    ('base_top','vertical_double_right'),('base_top','vertical_single_right'),('base_top','vertical_single_left'),('base_top','vertical_double_left')]

  for i in range(number_of_corners):
    corner_points.append(0)
  

  for i in range(number_of_corners):
    x1,y1,x2,y2 = court_lines_classificated[lines_to_intersect_for_corners[i][0]]
    x3,y3,x4,y4 = court_lines_classificated[lines_to_intersect_for_corners[i][1]]
    x,y = get_intersection_of_two_lines((x1,y1),(x2,y2),(x3,y3),(x4,y4))
    corner_points[i] = (int(x),int(y))

  if court_lines_classificated['net2'] == []:
    x1,y1,x2,y2 = court_lines_classificated['net1']
    x3,y3,x4,y4 = court_lines_classificated['vertical_double_left']
    x,y = get_intersection_of_two_lines((x1,y1),(x2,y2),(x3,y3),(x4,y4))
    corner_points_net.append((int(x),int(y)))
    x3,y3,x4,y4 = court_lines_classificated['vertical_double_right']
    x,y = get_intersection_of_two_lines((x1,y1),(x2,y2),(x3,y3),(x4,y4))
    corner_points_net.append((int(x),int(y)))
  else:
    x1,y1,x2,y2 = court_lines_classificated['net1'] 
    x3,y3,x4,y4 = court_lines_classificated['vertical_double_left']
    x,y = get_intersection_of_two_lines((x1,y1),(x2,y2),(x3,y3),(x4,y4)) 
    corner_points_net.append((int(x),int(y)))

    x3,y3,x4,y4 = court_lines_classificated['net2'] 
    px,py = get_intersection_of_two_lines((x1,y1),(x2,y2),(x3,y3),(x4,y4))
    corner_points_net.append((int(px),int(py)))

    x1,y1,x2,y2 = court_lines_classificated['net2'] 
    x3,y3,x4,y4 = court_lines_classificated['vertical_double_right']
    x,y = get_intersection_of_two_lines((x1,y1),(x2,y2),(x3,y3),(x4,y4))
    corner_points_net.append((int(x),int(y)))

  return corner_points, corner_points_net
def get_corner_points_for_tennis_court(base_line_len, margin_on_sides):
  """
    Description:
      Make artificial tennis court model and return corner and other specific points of that model
      parameters specifiyng size of the model, doesnt matter for homography, only if for visualization used
      For points number meaning look into documentation
    Parameters:
      base_line_len                       (int)                 : base horizontal line length in pixels
      margin_on_sides                     (int)                 : margin in pixels around tennis court model
    Returns:
      points_dst                          (list(list(int,int))  : points 0-13
      points_dst_without_center_line      (list(list(int,int))  : points 0-13 without point 5 and 8
      points_dst_all                      (list(list(int,int))  : points 0-20
  """


  height_to_width_ratio = (11.89*2 / 10.97)
  base_width = 200
  base_height = int(base_width * height_to_width_ratio)
  
  widht = base_width + margin_on_sides*2
  height = base_height + margin_on_sides*2

  
  ratio_two_vertical_lines_on_one_side = 1.37/10.97
  ratio_base_height_height_to_service_line = (5.49/(11.89*2))

  ratio_base_height_height_to_service_line = (5.49/(11.89*2))



  point1 = (widht - margin_on_sides, height - margin_on_sides)#

  point2 = (widht - ratio_two_vertical_lines_on_one_side * base_width - margin_on_sides, height - margin_on_sides)#
  point3 = (ratio_two_vertical_lines_on_one_side * base_width + margin_on_sides, height - margin_on_sides)#

  point4 = (0 + margin_on_sides, height - margin_on_sides)#

  point5 = (widht - ratio_two_vertical_lines_on_one_side * base_width - margin_on_sides, height - height*ratio_base_height_height_to_service_line - margin_on_sides)
  point6 = (margin_on_sides + base_width/2,height - height*ratio_base_height_height_to_service_line - margin_on_sides)
  point7 = (ratio_two_vertical_lines_on_one_side * base_width + margin_on_sides, height - height*ratio_base_height_height_to_service_line - margin_on_sides)

  point8 = (widht - ratio_two_vertical_lines_on_one_side * base_width - margin_on_sides, height*ratio_base_height_height_to_service_line + margin_on_sides)
  point9 = (margin_on_sides + base_width/2, height*ratio_base_height_height_to_service_line + margin_on_sides)
  point10 = (ratio_two_vertical_lines_on_one_side * base_width + margin_on_sides,height*ratio_base_height_height_to_service_line + margin_on_sides)

  point11 = (widht - margin_on_sides, 0 + margin_on_sides)#

  point12 = (widht - ratio_two_vertical_lines_on_one_side * base_width - margin_on_sides, 0 + margin_on_sides)#
  point13 = (ratio_two_vertical_lines_on_one_side * base_width + margin_on_sides, 0 + margin_on_sides)#

  point14 = (0 + margin_on_sides, 0 + margin_on_sides)#

  
  point15 =  (widht - ratio_two_vertical_lines_on_one_side * base_width - margin_on_sides, base_height/2 + margin_on_sides)
  point16 =  (base_width/2 + margin_on_sides, base_height/2 + margin_on_sides )
  point17 =  (ratio_two_vertical_lines_on_one_side * base_width + margin_on_sides, base_height/2 + margin_on_sides )

  point18 = (margin_on_sides + base_width/2, height - margin_on_sides)
  point19 = (margin_on_sides + base_width/2, 0 + margin_on_sides)
  
  point20 = (widht - margin_on_sides, base_height/2 + margin_on_sides)
  point21 = (0 + margin_on_sides, base_height/2 + margin_on_sides)

  points_dst = [[point1],[point2],[point3],[point4],[point5],[point6],[point7],[point8],[point9],[point10],[point11],[point12],[point13],[point14]]
  points_dst = np.array(points_dst)

  points_dst_without_center_line = [[point1],[point2],[point3],[point4],[point5],[point7],[point8],[point10],[point11],[point12],[point13],[point14]]
  points_dst_without_center_line = np.array(points_dst_without_center_line)

  points_dst_all = [[point1],[point2],[point3],[point4],[point5],[point6],[point7],[point8],[point9],[point10],[point11],[point12],[point13],[point14],[point15],[point16],[point17], [point18], [point19], [point20], [point21]]
  points_dst_all = np.array(points_dst_all)
  
  return points_dst,points_dst_without_center_line,points_dst_all
  
def find_homographies(corner_points_per_image, points_dst):
  """
    Description:
      Find homographies from court points detected in image and court points from artificial court model
    Parameters:
      corner_points_per_image                       (list(tuple(int,int)))               : court points found in image
      points_dst                                    (list(list(int,int))                 : court points of articifical court model
    Returns:
      homography                          (numpy.ndarray)  : homograhy matrix for projective transformation from image court to artificial court model, dimension 1x3x3
      homography_inv                      (numpy.ndarray)  : homograhy matrix for projective transformation from artificial court model to image court, dimension 1x3x3 
  """
  corner_points_per_image_formated = []
  for point in corner_points_per_image:
    corner_points_per_image_formated.append([point])
  corner_points_per_image_formated = np.array(corner_points_per_image_formated)
  homography, status = cv2.findHomography(corner_points_per_image_formated, points_dst)
  homography_inv,status_inv =  cv2.findHomography(points_dst, corner_points_per_image_formated)
  
  return homography,homography_inv
  
def prepare_images(images):
  """
    Description:
      Converts images from BGR to HSV
    Parameters:
      images                      (list((numpy.array(width,height, depth))))         : list of BGR images
    Returns:
      images_hsv                  (list((numpy.array(width,height, depth))))         : list of HSV images
  """
  images_hsv = []
  for image in images:
      img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      images_hsv.append(img_hsv)
      
  return images_hsv
  
def sort_by_first_component(item1, item2):
  return item1[0] - item2[0]
  
def get_specific_court_points_and_homographies(images, len_of_base_line_in_homography, margin_in_homography, image_heigth, image_width):
  """
    Description:
      Sequential version for finding court points and homographies
    Parameters:
      images                                              (list((numpy.array(width,height, depth))))         : list of BGR images
      len_of_base_line_in_homography                      (int)                                              : base horizontal line length in pixels for articifical court model
      margin_in_homography                                (int)                                              : margin in pixels around tennis court model for articifical court model
      image_heigth                                        (int)                                              : heigth of image in pixels
      image_width                                         (int)                                              : width of image in pixels
    Returns:
      corner_points_per_image                             (list(tuple(int,int)))                             : court points found in image                 
      court_found_per_image                               (list(bool))                                       : bool array with information if court was located or not
      homographies_image_to_artificial_court_per_frame    (list((numpy.ndarray)))                            : homograhy matrix for projective transformation from image court to artificial court model per image, dimension 1x3x3
      homographies_artificial_court_to_image_per_frame    (list((numpy.ndarray)))                            : homograhy matrix for projective transformation from artificial court model to image court per image, dimension 1x3x3 
  """
  images_hsv = prepare_images(images)
  
  histograms_for_images = {
        'hue':[],
        'saturation':[],
        'value':[],
    }
  for image in images_hsv:
    hist_h,hist_s,hist_v = calculate_hsv_histogram(image)
    histograms_for_images['hue'].append(hist_h)
    histograms_for_images['saturation'].append(hist_s)
    histograms_for_images['value'].append(hist_v)
  
  
  histogram_value_mode_climbing_steps = [140,100,60,30]
  histogram_saturation_mode_climbing_steps = [140,100,60,30]
  min_line_lengths = [int(min(image_heigth, image_width)/10), int(min(image_heigth, image_width)/5), int(min(image_heigth, image_width)/3)]
  v_s_climbing_steps = []
  for i in range(len(histogram_value_mode_climbing_steps)):
    for j in range(len(histogram_saturation_mode_climbing_steps)):
      for k in range(len(min_line_lengths)):
        v_s_climbing_steps.append((histogram_value_mode_climbing_steps[i],histogram_saturation_mode_climbing_steps[j],min_line_lengths[k]))
  
  
  
  
  masks = []
  corner_points_per_image = []
  corner_points_net_per_image = []
  court_lines_per_image = []
  court_found = True 
  court_found_per_image = []
  index_of_first_fail = 0
  continous_fails_tolerated = 10
  cnt_continous_fails = 0


  for i in range(len(images_hsv)):
    if court_found == False:
      cnt_continous_fails = cnt_continous_fails + 1
    else:
      cnt_continous_fails = 0
    if court_found == False and cnt_continous_fails >= continous_fails_tolerated: #skip similiar images
     value_sim = cv2.compareHist( histograms_for_images['value'][index_of_first_fail], histograms_for_images['value'][i], cv2.HISTCMP_CORREL )
     saturation_sim = cv2.compareHist( histograms_for_images['saturation'][index_of_first_fail], histograms_for_images['saturation'][i], cv2.HISTCMP_CORREL )
     if value_sim >= 0.9 and saturation_sim >= 0.9:
       print(f"{i} skip by hist sim")
       continue
    court_found = False
    for j in range(len(v_s_climbing_steps)):
      bins_to_climb_saturation = v_s_climbing_steps[j][0]
      bins_to_climb_value = v_s_climbing_steps[j][1]
      min_line_length = v_s_climbing_steps[j][2]
      
      lines_count_threshold = int((image_heigth/min_line_length) * 4 + (image_width/min_line_length) * 6 + ((image_heigth/min_line_length) * 4 + (image_width/min_line_length) * 6) * 0.3)
      
      image = images_hsv[i]
      hist_v = histograms_for_images['value'][i]
      hist_s = histograms_for_images['saturation'][i]
      mask = get_mask_of_court_lines(image, hist_v, hist_s,bins_to_climb_value,bins_to_climb_saturation)
      lines_per_mask = get_lines_from_mask(mask, min_line_length)
      if lines_per_mask is None:
        continue
      if len(lines_per_mask) > lines_count_threshold: #too many lines detected
        continue
      horizontal_lines, vertical_lines = find_horizontal_and_vertical_lines(lines_per_mask)
      if horizontal_lines is None or vertical_lines is None:
        continue
        
      court_lines = classify_court_lines(horizontal_lines, vertical_lines)
      corner_points, corner_points_net = find_court_corner_points(court_lines)
      corner_points_per_image.append(corner_points)
      corner_points_net_per_image.append(corner_points_net)
      court_lines_per_image.append([horizontal_lines,vertical_lines])
      masks.append(mask)
      if j != 0:
        move_forward = v_s_climbing_steps[j]
        v_s_climbing_steps.pop(j)
        v_s_climbing_steps.insert(0,move_forward)
      court_found = True
      print(f"image {i} court found {court_found}")
      break
    if not court_found:
      index_of_first_fail = i
    court_found_per_image.append(court_found)
    
    
    points_dst,points_dst_without_center_line, points_dst_all = get_corner_points_for_tennis_court(len_of_base_line_in_homography, margin_in_homography)
    
    
    homographies_image_to_artificial_court_per_frame = []
    homographies_artificial_court_to_image_per_frame = []
    homography = None
    homography_inv = None
    for i in range(len(corner_points_per_image)):
      if len(corner_points_per_image[i]) == 14:
        homography,homography_inv = find_homographies(corner_points_per_image[i], points_dst)
      elif len(corner_points_per_image[i]) == 12:
        homography,homography_inv = find_homographies(corner_points_per_image[i], points_dst_without_center_line)

      homographies_image_to_artificial_court_per_frame.append(homography)
      homographies_artificial_court_to_image_per_frame.append(homography_inv)
      
      
  point6 = points_dst_all[5]
  point9 = points_dst_all[8]
  point15 = points_dst_all[14]
  point16 = points_dst_all[15]
  point17 = points_dst_all[16]
  point18 = points_dst_all[17]
  point19 = points_dst_all[18]
  point20 = points_dst_all[19]
  point21 = points_dst_all[20]
  points = np.array([point6,point9, point15, point16, point17, point18, point19, point20, point21])
  for i in range(len(corner_points_per_image)):
    transformed = cv2.perspectiveTransform(points, homographies_artificial_court_to_image_per_frame[i])
    if len(corner_points_per_image[i]) == 12: 
      corner_points_per_image[i].insert(5,(int(transformed[0][0][0]),int(transformed[0][0][1])))
      corner_points_per_image[i].insert(8,(int(transformed[1][0][0]),int(transformed[1][0][1])))
    corner_points_per_image[i].append((int(transformed[2][0][0]),int(transformed[2][0][1])))
    corner_points_per_image[i].append((int(transformed[3][0][0]),int(transformed[3][0][1])))
    corner_points_per_image[i].append((int(transformed[4][0][0]),int(transformed[4][0][1])))
    corner_points_per_image[i].append((int(transformed[5][0][0]),int(transformed[5][0][1])))
    corner_points_per_image[i].append((int(transformed[6][0][0]),int(transformed[6][0][1])))
    corner_points_per_image[i].append((int(transformed[7][0][0]),int(transformed[7][0][1])))
    corner_points_per_image[i].append((int(transformed[8][0][0]),int(transformed[8][0][1])))
    if len(corner_points_net_per_image[i]) == 2:
      corner_points_per_image[i].append(corner_points_net_per_image[i][0])
      corner_points_per_image[i].append((int((corner_points_net_per_image[i][0][0] + corner_points_net_per_image[i][1][0])/2) , int((corner_points_net_per_image[i][0][1] + corner_points_net_per_image[i][1][1])/2)))
      corner_points_per_image[i].append(corner_points_net_per_image[i][1])
    else:
      corner_points_per_image[i].append(corner_points_net_per_image[i][0])
      corner_points_per_image[i].append(corner_points_net_per_image[i][1])
      corner_points_per_image[i].append(corner_points_net_per_image[i][2])
      
      
  for i in range(len(court_found_per_image)):
    if court_found_per_image[i] == False:
      corner_points_per_image.insert(i,None)
      
  return corner_points_per_image, court_found_per_image, homographies_image_to_artificial_court_per_frame, homographies_artificial_court_to_image_per_frame
  
  
  
def get_specific_court_points_and_homographies_parallel(images, len_of_base_line_in_homography, margin_in_homography, image_heigth, image_width, tasks_to_create, pool):
  """
    Description:
      Parallel version for finding court points and homographies
    Parameters:
      images                                              (list((numpy.array(width,height, depth))))         : list of BGR images
      len_of_base_line_in_homography                      (int)                                              : base horizontal line length in pixels for articifical court model
      margin_in_homography                                (int)                                              : margin in pixels around tennis court model for articifical court model
      image_heigth                                        (int)                                              : heigth of image in pixels
      image_width                                         (int)                                              : width of image in pixels
      tasks_to_create                                     (int)                                              : tasks to create for parallel processing
      pool                                                (multiprocessing.pool)                             : pool of processes for whom tasks will be assigned
    Returns:
      corner_points_per_image                             (list(tuple(int,int)))                             : court points found in image                 
      court_found_per_image                               (list(bool))                                       : bool array with information if court was located or not
      homographies_image_to_artificial_court_per_frame    (list((numpy.ndarray)))                            : homograhy matrix for projective transformation from image court to artificial court model per image, dimension 1x3x3
      homographies_artificial_court_to_image_per_frame    (list((numpy.ndarray)))                            : homograhy matrix for projective transformation from artificial court model to image court per image, dimension 1x3x3 
  """

  
  num_images = len(images)
  with Manager() as manager:
    intervals = manager.list()
    interval_len = num_images / tasks_to_create
    for i in range(tasks_to_create):
      if i == (tasks_to_create-1):
        intervals.append((int(i*interval_len), int(num_images)))
      else:
        intervals.append((int(i*interval_len), int((i+1)*interval_len)))
        
    #shared memory manually, because processes do not share memory
    corner_points_per_image = manager.list()
    court_found_per_image = manager.list()
    homographies_image_to_artificial_court_per_frame = manager.list()
    homographies_artificial_court_to_image_per_frame = manager.list()
    for i in range(len(images)):
      corner_points_per_image.append([])
      court_found_per_image.append([])
      homographies_image_to_artificial_court_per_frame.append([])
      homographies_artificial_court_to_image_per_frame.append([])
    
    images = manager.list(images)

    functions_list = []
    for i in range(tasks_to_create):
      #prepare tasks
      functions_list.append(functools.partial(get_specific_court_points_and_homographies2, images, len_of_base_line_in_homography, margin_in_homography, image_heigth,\
      image_width,corner_points_per_image, court_found_per_image, homographies_image_to_artificial_court_per_frame, homographies_artificial_court_to_image_per_frame, intervals, i))
    #run tasks
    res = pool.map(smap, functions_list)
    #convert back to lists from shared memory
    images = list(images) 
    corner_points_per_image = list(corner_points_per_image)
    court_found_per_image = list(court_found_per_image)
    homographies_image_to_artificial_court_per_frame = list(homographies_image_to_artificial_court_per_frame)
    homographies_artificial_court_to_image_per_frame = list(homographies_artificial_court_to_image_per_frame)
    
    return corner_points_per_image, court_found_per_image, homographies_image_to_artificial_court_per_frame, homographies_artificial_court_to_image_per_frame
    
    
    
    
def smap(f):
  """
    Description:
      helper function for multiprocessing
  """
  return f()  
    
    
    
def get_specific_court_points_and_homographies2(images, len_of_base_line_in_homography, margin_in_homography, image_heigth, image_width, corner_points_per_image_res,\
court_found_per_image_res, homographies_image_to_artificial_court_per_frame_res, homographies_artificial_court_to_image_per_frame_res,intervals, id):
  """
    Description:
      Task for process in parallel processing version
    Parameters:
      images                                                  (list((numpy.array(width,height, depth))))         : list of BGR images
      len_of_base_line_in_homography                          (int)                                              : base horizontal line length in pixels for articifical court model
      margin_in_homography                                    (int)                                              : margin in pixels around tennis court model for articifical court model
      image_heigth                                            (int)                                              : heigth of image in pixels
      image_width                                             (int)                                              : width of image in pixels
      corner_points_per_image_res                             (list())                                           : list to which insert local results
      court_found_per_image_res                               (list())                                           : list to which insert local results
      homographies_image_to_artificial_court_per_frame_res    (list())                                           : list to which insert local results
      homographies_artificial_court_to_image_per_frame_res    (list())                                           : list to which insert local results
      intervals                                               (int,int)                                          : interval of images to process
      id                                                      (int)                                              : process id
    Returns:
      
  """


  start = intervals[id][0]
  end = intervals[id][1]

  histogram_value_mode_climbing_steps = [140,100,60,30]
  histogram_saturation_mode_climbing_steps = [140,100,60,30]
  min_line_lengths = [int(min(image_heigth, image_width)/10), int(min(image_heigth, image_width)/5), int(min(image_heigth, image_width)/3)]
  v_s_climbing_steps = []
  for i in range(len(histogram_value_mode_climbing_steps)):
    for j in range(len(histogram_saturation_mode_climbing_steps)):
      for k in range(len(min_line_lengths)):
        v_s_climbing_steps.append((histogram_value_mode_climbing_steps[i],histogram_saturation_mode_climbing_steps[j],min_line_lengths[k]))
  
  
  
  
  masks = []
  corner_points_per_image = []
  corner_points_net_per_image = []
  court_lines_per_image = []
  court_found = True 
  court_found_per_image = []
  index_of_first_fail = 0
  continous_fails_tolerated = 10
  cnt_continous_fails = 0


  histogram_failed_image_hue = []
  histogram_failed_image_saturation = []
  histogram_failed_image_value = []
  
  
  
  for i in range(start, end):
    image = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
    hist_h,hist_s,hist_v = calculate_hsv_histogram(image)
    
    if court_found == False:
      cnt_continous_fails = cnt_continous_fails + 1
    else:
      cnt_continous_fails = 0
    if court_found == False and cnt_continous_fails >= continous_fails_tolerated: 
     value_sim = cv2.compareHist( histogram_failed_image_value, hist_v, cv2.HISTCMP_CORREL )
     saturation_sim = cv2.compareHist( histogram_failed_image_saturation, hist_s, cv2.HISTCMP_CORREL )
     if value_sim >= 0.9 and saturation_sim >= 0.9:
       print(f"{i} skip by hist sim")
       continue
    court_found = False

    for j in range(len(v_s_climbing_steps)):
      bins_to_climb_saturation = v_s_climbing_steps[j][0]
      bins_to_climb_value = v_s_climbing_steps[j][1]
      
      min_line_length = v_s_climbing_steps[j][2]
      lines_count_threshold = int((image_heigth/min_line_length) * 4 + (image_width/min_line_length) * 6 + ((image_heigth/min_line_length) * 4 + (image_width/min_line_length) * 6) * 0.3)
      

      mask = get_mask_of_court_lines(image, hist_v, hist_s,bins_to_climb_value,bins_to_climb_saturation)
      lines_per_mask = get_lines_from_mask(mask, min_line_length)
      if lines_per_mask is None:
        continue
      if len(lines_per_mask) > lines_count_threshold: #too many lines detected
        continue
      horizontal_lines, vertical_lines = find_horizontal_and_vertical_lines(lines_per_mask)
      if horizontal_lines is None or vertical_lines is None:
        continue
      court_lines = classify_court_lines(horizontal_lines, vertical_lines)
      corner_points, corner_points_net = find_court_corner_points(court_lines)
      corner_points_per_image.append(corner_points)
      corner_points_net_per_image.append(corner_points_net)
      court_lines_per_image.append([horizontal_lines,vertical_lines])
      masks.append(mask)
      if j != 0:
        move_forward = v_s_climbing_steps[j]
        v_s_climbing_steps.pop(j)
        v_s_climbing_steps.insert(0,move_forward)
      court_found = True
      break
    if not court_found:
      histogram_failed_image_hue = histogram_current_image_hue
      histogram_failed_image_saturation = histogram_current_image_saturation
      histogram_failed_image_value = histogram_current_image_value
    print(f"image {i} court found {court_found}")
    court_found_per_image.append(court_found)
    
    
    points_dst,points_dst_without_center_line, points_dst_all = get_corner_points_for_tennis_court(len_of_base_line_in_homography, margin_in_homography)
    
    
    homographies_image_to_artificial_court_per_frame = []
    homographies_artificial_court_to_image_per_frame = []
    homography = None
    homography_inv = None
    for i in range(len(corner_points_per_image)):
      if len(corner_points_per_image[i]) == 14:
        homography,homography_inv = find_homographies(corner_points_per_image[i], points_dst)
      elif len(corner_points_per_image[i]) == 12:
        homography,homography_inv = find_homographies(corner_points_per_image[i], points_dst_without_center_line)

      homographies_image_to_artificial_court_per_frame.append(homography)
      homographies_artificial_court_to_image_per_frame.append(homography_inv)
      
      
  point6 = points_dst_all[5]
  point9 = points_dst_all[8]
  point15 = points_dst_all[14]
  point16 = points_dst_all[15]
  point17 = points_dst_all[16]
  point18 = points_dst_all[17]
  point19 = points_dst_all[18]
  point20 = points_dst_all[19]
  point21 = points_dst_all[20]
  points = np.array([point6,point9, point15, point16, point17, point18, point19, point20, point21])
  for i in range(len(corner_points_per_image)):
    transformed = cv2.perspectiveTransform(points, homographies_artificial_court_to_image_per_frame[i])
    if len(corner_points_per_image[i]) == 12: 
      corner_points_per_image[i].insert(5,(int(transformed[0][0][0]),int(transformed[0][0][1])))
      corner_points_per_image[i].insert(8,(int(transformed[1][0][0]),int(transformed[1][0][1])))
    corner_points_per_image[i].append((int(transformed[2][0][0]),int(transformed[2][0][1])))
    corner_points_per_image[i].append((int(transformed[3][0][0]),int(transformed[3][0][1])))
    corner_points_per_image[i].append((int(transformed[4][0][0]),int(transformed[4][0][1])))
    corner_points_per_image[i].append((int(transformed[5][0][0]),int(transformed[5][0][1])))
    corner_points_per_image[i].append((int(transformed[6][0][0]),int(transformed[6][0][1])))
    corner_points_per_image[i].append((int(transformed[7][0][0]),int(transformed[7][0][1])))
    corner_points_per_image[i].append((int(transformed[8][0][0]),int(transformed[8][0][1])))
    if len(corner_points_net_per_image[i]) == 2:
      corner_points_per_image[i].append(corner_points_net_per_image[i][0])
      corner_points_per_image[i].append((int((corner_points_net_per_image[i][0][0] + corner_points_net_per_image[i][1][0])/2) , int((corner_points_net_per_image[i][0][1] + corner_points_net_per_image[i][1][1])/2)))
      corner_points_per_image[i].append(corner_points_net_per_image[i][1])
    else:
      corner_points_per_image[i].append(corner_points_net_per_image[i][0])
      corner_points_per_image[i].append(corner_points_net_per_image[i][1])
      corner_points_per_image[i].append(corner_points_net_per_image[i][2])
      
      
  for i in range(len(court_found_per_image)):
    if court_found_per_image[i] == False:
      corner_points_per_image.insert(i,None)
  for i in range(start,end):
    corner_points_per_image_res[i] = corner_points_per_image[i-start]
    court_found_per_image_res[i] = court_found_per_image[i-start]
    homographies_image_to_artificial_court_per_frame_res[i] = homographies_image_to_artificial_court_per_frame[i-start]
    homographies_artificial_court_to_image_per_frame_res[i] = homographies_artificial_court_to_image_per_frame[i-start]
  
  
  
  