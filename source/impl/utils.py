import numpy as np
import math
from functools import cmp_to_key
import os
import cv2
import scipy
import sys
sys.path.append('.')
from yolo3.utils.bbox import BoundBox


def linear_interpolation_of_trajectory(trajectory):
  """
    Description:
      Linear interpolation of mssing detections in trajectory
    Parameters:
      trajectory         (list(int,int,int))                                 : trajectory represented as frame,x,y       
    Returns:
      trajectory         (list(int,int,int))                                 : trajectory with interpolated missing detections, represented as frame,x,y       
  """

  intervals = []
  for i in range(len(trajectory)-1):
    if (trajectory[i][0]+1) != trajectory[i+1][0]:
      intervals.append((trajectory[i][0],trajectory[i+1][0],i,i+1))


  for j in range(len(intervals)-1,-1,-1):
    #intervals start_frame, end_frame,Start_traj_id,end_traj_id
    interval = intervals[j]
    num_missing = interval[1] - interval[0] - 1
    x_distance_between_detections = trajectory[interval[3]][1] - trajectory[interval[2]][1]
    y_distance_between_detections = trajectory[interval[3]][2] - trajectory[interval[2]][2]
    x_shift = x_distance_between_detections / (num_missing+1)
    y_shift = y_distance_between_detections / (num_missing+1)
    x_start = trajectory[interval[2]][1]
    y_start = trajectory[interval[2]][2]
    cnt = 0

    for i in range(num_missing-1,-1,-1):
      cnt = cnt + 1
      trajectory.insert(interval[3],(interval[0]+i+1,x_start + (i+1) * x_shift,y_start + (i+1) * y_shift))
  return trajectory
  
def compare_file_paths(path1, path2):
  """
    Compare file paths with filenames made by numbers.

    Attributes:
      path1	(str)	path to one file
      path2	(str)	path to second file
    Returns
          (int)	>0 for path1 has bigger number as filename, 0 for equal numbers in filename, <0 for path2 has bigger number as filename
  """
  path1 = path1.replace('\\','/')
  path2 = path2.replace('\\','/')
  path1 = path1.rsplit('/', 1)[-1]
  path1 = path1.rsplit('.', 1)[0]
  path1 = int(path1)
  path2 = path2.rsplit('/', 1)[-1]
  path2 = path2.rsplit('.', 1)[0]
  path2 = int(path2)

  return path1-path2
  
def save_frames_in_time_interval_from_video(videopath, dir_to_save_path, frame_number_start, frame_number_end):
  """
    Description:
      Save specified frames from video
    Parameters:
      videopath                 (string)        : path to video     
      dir_to_save_path          (string)        : directory where to save frames   
      frame_number_start        (int)           : first frame  
      frame_number_end          (int)           : last frame       
    Returns:
  """
  isExist =  os.path.exists(dir_to_save_path)
  if not isExist:
    os.makedirs(dir_to_save_path)
    cap = cv2.VideoCapture(videopath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_start)
    PATH_TO_SAVE_FRAMES = dir_to_save_path
    count = 0
    for i in range(frame_number_start,frame_number_end+1):
      res, frame = cap.read()#read posouva frame ukazatel sam o 1
      cv2.imwrite(f"{PATH_TO_SAVE_FRAMES}{count}.jpg", frame)
      print(i)
      count = count + 1
  else:
    print(f'dir {dir_to_save_path} already exist')
    
def save_images_as_video(images, PATH_TO_SAVE_VIDEO,fps):
  """
    Description:
      Save images as video
    Parameters:
      images                   (list((numpy.array(width,height, depth))))        : list of BGR images
      PATH_TO_SAVE_VIDEO       (string)                                          : path to save video     
      fps                      (int)                                             : desired fps of saved video  
    Returns:
  """
  out = cv2.VideoWriter(PATH_TO_SAVE_VIDEO,cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), fps, (images[0].shape[1],images[0].shape[0])) 
  for i in range(len(images)):
      out.write(images[i])
  out.release()    
  
  
def load_frames_in_time_interval_from_video(videopath, frame_number_start, frame_number_end):
  """
    Description:
      Load specified frames from video and get fps information
    Parameters:
      videopath                 (string)                                        : path to video     
      frame_number_start        (int)                                           : first frame  
      frame_number_end          (int)                                           : last frame       
    Returns:
      frames                    (list((numpy.array(width,height, depth))))      : list of BGR images
      int(fps)                  (int)                                           : fps of video
  """
  frames = []
  cap = cv2.VideoCapture(videopath)
  fps = cap.get(cv2.CAP_PROP_FPS)
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_start)
  for i in range(frame_number_start,frame_number_end+1):
    res, frame = cap.read()#read shift by one automatically
    if frame is None:
      return None,None
    frames.append(frame)
    print(i)
  return frames,int(fps)

    
  
class Path:
  """
	Class for representing trajectories in video

    Class for reprenting trajectories in video. Trajectories are represented as coordinates x,y of detections with information about frame where detections were located.

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

def get_angle2(point1, point2, point3): 
  """
    Description:
      Computes angle between vectors. Vector1 is vector of direction between point1 and point2. Vector2 is vector of direction between point2 and point3.
    Parameters:
      point1       (tuple[int,int]) : coordinates x-y of point
      point2       (tuple[int,int]) : coordinates x-y of point
      point3       (tuple[int,int]) : coordinates x-y of point
    Returns:
      angle        (int)            : angle, see desc.
  """
  vector_1 = [point2[0] - point1[0], point2[1] - point1[1]]
  vector_2 = [point3[0] - point2[0], point3[1] - point2[1]]
  if np.linalg.norm(vector_1) == 0 or np.linalg.norm(vector_2) == 0:
    return 0

  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)

  dot_product = np.dot(unit_vector_1, unit_vector_2)
  if dot_product > 1:
    dot_product = 1
  elif dot_product < -1:
    dot_product = -1 
  angle = np.arccos(dot_product) *  180 / np.pi 

  return angle
  
def get_angle_vectors(vector_1, vector_2 ):
  """
    Description:
      Computes angle between vectors. Vector1 is vector of direction between point1 and point2. Vector2 is vector of direction between point2 and point3.
      
    Parameters:
      vector_1       (tuple[int,int]) : x,y 
      vector_2       (tuple[int,int]) : x,y 
    Returns:
      angle        (int)                : angle
  """

  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)

  if dot_product > 1:
    dot_product = 1
  elif dot_product < -1:
    dot_product = -1 
  angle = np.arccos(dot_product) *  180 / np.pi 

  return angle  

def get_angle(point1,point2,point3):
  """
    Description:
      Computes angle between vectors. Vector1 is vector of direction between point1 and point2. Vector2 is vector of direction between point2 and point3.
    Parameters:
      point1       (tuple[int,int,int]) : frame,x,y 
      point2       (tuple[int,int,int]) : frame,x,y 
      point3       (tuple[int,int,int]) : frame,x,y 
    Returns:
      angle        (int)                : angle
  """
  vector_1 = [point2[1] - point1[1], point2[2] - point1[2]]
  vector_2 = [point3[1] - point2[1], point3[2] - point2[2]]

  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)
  if dot_product > 1:
    dot_product = 1
  elif dot_product < -1:
    dot_product = -1 
  angle = np.arccos(dot_product) *  180 / np.pi 
  return angle

def rotate_vector(vector, angle):
  """
    Description:
      Rotates vector by angle.
    Parameters:
      vector         (tuple[float,float]): vector to be rotated
      angle          (int)               : angle for rotation, 0-360
    Returns:
      rotated_vector (tuple[float,float]): rotated vector
  """
  rad = angle * np.pi/180
  sin = np.sin(rad)
  cos = np.cos(rad)
  rotated_vector = ( cos * vector[0] +  sin * vector[1], - sin * vector[0] + cos * vector[1])
  return rotated_vector  

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

def get_intersection_of_two_lines(start1,end1,start2,end2):
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

  return (px,py) 
  
def compute_distance_abs(point1, point2):
  """
    Description:
      Computes L2 distance between two points.
    Parameters:
       point1 (tuple[int,int]): coordinates x-y of first point
       point2 (tuple[int,int]): coordinates x-y of second point

    Returns:
              (float)         : L2 distance between point1 and point2
    """
  sum = 0
  for i in range(2):
    sum = sum + abs(point1[i] - point2[i])
  return sum
    
  
def compute_distance(point1, point2):
  """
    Description:
      Computes L2 distance between two points.
    Parameters:
       point1 (tuple[int,int]): coordinates x-y of first point
       point2 (tuple[int,int]): coordinates x-y of second point

    Returns:
              (float)         : L2 distance between point1 and point2
    """
  sum = 0
  for i in range(2):
    sum = sum + (point1[i] - point2[i]) * (point1[i] - point2[i])
  return math.sqrt(sum)
  #return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]) )

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
  #20 na kazdou stranu jako okraj
  widht = base_width + margin_on_sides*2
  height = base_height + margin_on_sides*2

  #3 a 4
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

  points_dst_all = [point1,point2,point3,point4,point5,point6,point7,point8,point9,point10,point11,point12,point13,point14,point15,point16,point17,point18,point19,point20, point21]

  return points_dst_all  
  
  
def is_inside_area2(position, bottom_right_corner, bottom_left_corner, top_right_corner, top_left_corner, image_width, image_height):
  """
    Description:
      if position inside specified rectangle area, line left should intersect only one vertical and line up should intersect only one horizontal line if inside area
    Parameters:
      position                        ((int,int))                         : player position, x,y
      bottom_right_corner             (int)                               : bottom_right_corner of area of interest
      bottom_left_corner              (int)                               : bottom_left_corner of area of interest
      top_right_corner                (int)                               : top_right_corner of area of interest
      top_left_corner                 (int)                               : top_left_corner of area of interest
      image_width                     (int)                               : image width in pixels
      image_height                    (int)                               : image heigth in pixels
    Returns:
                                      (bool)                              : True if position inside area
  """
  vertical_vector_upwards1 = (top_right_corner[0] - bottom_right_corner[0],top_right_corner[1] - bottom_right_corner[1])
  vertical_vector_upwards2 = (top_left_corner[0] - bottom_left_corner[0],top_left_corner[1] - bottom_left_corner[1])

  horizontal_vector_rightwards1 = (bottom_right_corner[0] - bottom_left_corner[0],bottom_right_corner[1] - bottom_left_corner[1])
  horizontal_vector_rightwards2 = (top_right_corner[0] - top_left_corner[0],top_right_corner[1] - top_left_corner[1])

  end_of_line_to_right_from_player_vector1 = get_end_point_from_vector_and_point(horizontal_vector_rightwards1,position,image_width,image_height)
  end_of_line_to_right_from_player_vector2 = get_end_point_from_vector_and_point(horizontal_vector_rightwards2,position,image_width,image_height)

  end_of_line_up_from_player_vector1 = get_end_point_from_vector_and_point(vertical_vector_upwards1,position,image_width,image_height)
  end_of_line_up_from_player_vector2 = get_end_point_from_vector_and_point(vertical_vector_upwards2,position,image_width,image_height)





  intersection_player_with_left_vertical_x_1,intersection_player_with_left_vertical_y_1 =         get_intersection_of_two_lines_segments(position ,end_of_line_to_right_from_player_vector1,bottom_left_corner ,top_left_corner)
  intersection_player_with_right_vertical_x_1,intersection_player_with_right_vertical_y_1 =       get_intersection_of_two_lines_segments(position ,end_of_line_to_right_from_player_vector1,bottom_right_corner ,top_right_corner)
  
  intersection_player_with_left_vertical_x_2,intersection_player_with_left_vertical_y_2 =         get_intersection_of_two_lines_segments(position ,end_of_line_to_right_from_player_vector2,bottom_left_corner ,top_left_corner)
  intersection_player_with_right_vertical_x_2,intersection_player_with_right_vertical_y_2 =       get_intersection_of_two_lines_segments(position ,end_of_line_to_right_from_player_vector2,bottom_right_corner ,top_right_corner)
  
  

  intersection_player_with_top_horizontal_x_1,intersection_player_with_top_horizontal_y_1 =       get_intersection_of_two_lines_segments(position ,end_of_line_up_from_player_vector1,bottom_left_corner ,bottom_right_corner)
  intersection_player_with_bottom_horizontal_x_1,intersection_player_with_bottom_horizontal_y_1 = get_intersection_of_two_lines_segments(position ,end_of_line_up_from_player_vector1,top_left_corner ,top_right_corner)

  intersection_player_with_top_horizontal_x_2,intersection_player_with_top_horizontal_y_2 =       get_intersection_of_two_lines_segments(position ,end_of_line_up_from_player_vector2,bottom_left_corner ,bottom_right_corner)
  intersection_player_with_bottom_horizontal_x_2,intersection_player_with_bottom_horizontal_y_2 = get_intersection_of_two_lines_segments(position ,end_of_line_up_from_player_vector2,top_left_corner ,top_right_corner)



  is_vertical_intersection_left_1 = ((intersection_player_with_left_vertical_x_1 != -1) and (intersection_player_with_left_vertical_y_1 != -1))
  is_vertical_intersection_right_1 = ((intersection_player_with_right_vertical_x_1 != -1) and (intersection_player_with_right_vertical_x_1 != -1))

  is_vertical_intersection_left_2 = ((intersection_player_with_left_vertical_x_2 != -1) and (intersection_player_with_left_vertical_y_2 != -1))
  is_vertical_intersection_right_2 = ((intersection_player_with_right_vertical_x_2 != -1) and (intersection_player_with_right_vertical_y_2 != -1))


  is_horizontal_intersection_top_1 = ((intersection_player_with_top_horizontal_x_1 != -1) and (intersection_player_with_top_horizontal_y_1 != -1))
  is_horizontal_intersection2_bottom_1 = ((intersection_player_with_bottom_horizontal_x_1 != -1) and (intersection_player_with_bottom_horizontal_y_1 != -1))

  is_horizontal_intersection_top_2 = ((intersection_player_with_top_horizontal_x_2 != -1) and (intersection_player_with_top_horizontal_y_2 != -1))
  is_horizontal_intersection_bottom_2 = ((intersection_player_with_bottom_horizontal_x_2 != -1) and (intersection_player_with_bottom_horizontal_y_2 != -1))




  if not ( ((is_vertical_intersection_left_1 != is_vertical_intersection_right_1) or (is_vertical_intersection_left_2 != is_vertical_intersection_right_2)) and \
((is_horizontal_intersection_top_1 != is_horizontal_intersection2_bottom_1) or (is_horizontal_intersection_top_2 != is_horizontal_intersection_bottom_2)) ):
    return 'out'
  else:
    return 'in'




def is_inside_area(position, bottom_right_corner, bottom_left_corner, top_right_corner, top_left_corner, image_width, image_height):
  """
    Description:
      if position inside specified convex polygon aree
        #code from: https://matlabgeeks.com/tips-tutorials/computational-geometry/determine-if-a-point-is-located-within-a-convex-polygon/
    Parameters:
      position                        ((int,int))                         : player position, x,y
      bottom_right_corner             (int)                               : bottom_right_corner of area of interest
      bottom_left_corner              (int)                               : bottom_left_corner of area of interest
      top_right_corner                (int)                               : top_right_corner of area of interest
      top_left_corner                 (int)                               : top_left_corner of area of interest
      image_width                     (int)                               : image width in pixels
      image_height                    (int)                               : image heigth in pixels
    Returns:
                                      (bool)                              : True if position inside area
  """
  corners_clock_wise = [(bottom_left_corner[0], bottom_left_corner[1], 0),(top_left_corner[0], top_left_corner[1], 0),(top_right_corner[0], top_right_corner[1], 0),(bottom_right_corner[0], bottom_right_corner[1], 0)]
  position = (position[0],position[1],0)
  insidePoly = True
  number_of_corners = 4

  for k in range(number_of_corners):
    # determine vectors for each segment around boundary
    # starting between p2-p1, p3-p2, etc.        
    # if at the final node, the vector connects back to p1
    if k == (number_of_corners-1):
        current_poly = (corners_clock_wise[0][0] - corners_clock_wise[-1][0],corners_clock_wise[0][1] - corners_clock_wise[-1][1], 0)
    else:
        current_poly = (corners_clock_wise[k+1][0] - corners_clock_wise[k][0],corners_clock_wise[k+1][1] - corners_clock_wise[k][1], 0)

    #vector from point in space vertex on polygon (point-p(k))
    point_vect = (position[0] - corners_clock_wise[k][0], position[1] - corners_clock_wise[k][1], 0);

    # determine if the point is inside or outside the polygon:
    # if cross product of all polygon_vectors x point_vectors is
    # negative then the point is inside (if polygon is convex) 
    # take cross_product of point vector and polygon vector       
    c = np.cross(current_poly, point_vect);
    current_sign = np.sign(c[2]);
    if k == 0:
       check_sign = current_sign;
    elif check_sign != current_sign:
        insidePoly = False;
  if insidePoly:
    return 'in'
  else:
    return 'out'
  
def is_inside_middle_lines_extended_area(ball_position,corner_points_per_images,current_frame_index,image_width, image_height):
  """
    Description:
      If detections is in area of extended to end of image of two vertical serve lines 
    Parameters:
      ball_position                   ((int,int))                         : player position, x,y
      corner_points_per_images        (list(tuple(int,int)))              : court points found in images
      current_frame_index             (int)                               : frame index of detection
      image_width                     (int)                               : image width in pixels
      image_height                    (int)                               : image heigth in pixels
    Returns:
                                      (bool)                              : True if position inside area
  """

  left_top = corner_points_per_images[current_frame_index][9]
  right_top = corner_points_per_images[current_frame_index][7]

  left_bottom = corner_points_per_images[current_frame_index][6]
  right_bottom = corner_points_per_images[current_frame_index][4]
  
  vector_left1 = (left_top[0] - right_top[0], left_top[1] - right_top[1])
  vector_right1 = (right_top[0] - left_top[0], right_top[1] - left_top[1])

  vector_left2 = (left_bottom[0] - right_bottom[0], left_bottom[1] - right_bottom[1])
  vector_right2 = (right_bottom[0] - left_bottom[0], right_bottom[1] - left_bottom[1])

  left_end1 = get_end_point_from_vector_and_point(vector_left1, left_top, image_width, image_height)
  right_end1 = get_end_point_from_vector_and_point(vector_right1, right_top, image_width, image_height)



  left_end2 = get_end_point_from_vector_and_point(vector_left2, left_bottom, image_width, image_height)
  right_end2 = get_end_point_from_vector_and_point(vector_right2, right_bottom, image_width, image_height)

  inside_middle_lines_extended_area = is_inside_area(ball_position, right_end2, left_end2, right_end1, left_end1, image_width, image_height)

  return inside_middle_lines_extended_area    

def get_end_point_from_vector_and_point(vector, start_point, img_width, img_height):
  """
    Description:
      Get second point of line starting at start_point and has direction of vector on image with shape img_width x img_height.
      The second point is on an edge of the image.
    Parameters:
      vector       (tuple(int,int)) : vector of direction of line
      start_point  (tuple(int,int)) : one end point of line
      img_width    (int)            : width of image
      img_height   (int)            : height of image
    Returns:
      (x,y)        (tuple[int,int]) : second end point of line
  """

  if vector == (0,0):
    print(f"zero vector in get_end_point_from_vector_and_point")
    return (0,0)
  x_steps = 0
  y_steps = 0
  x = -1
  y = -1
  if vector[0] > 0:
    x_steps = (img_width-start_point[0])/vector[0]
  elif vector[0] < 0:
    x_steps = - (start_point[0]/vector[0])
  elif vector[0] == 0: 
    x_steps = 0

  if vector[1] > 0:
    y_steps = (img_height-start_point[1])/vector[1]
  elif vector[1] < 0:
    y_steps = - (start_point[1]/vector[1])
  elif vector[1] == 0: 
    y_steps = 0

  if y_steps == 0:
    y_steps = float('inf')
  if x_steps == 0:
    x_steps = float('inf')

  if x_steps > y_steps:
    x = start_point[0] + y_steps * vector[0] 
    y = start_point[1] + y_steps * vector[1]
  else:
    x = start_point[0] + x_steps * vector[0]
    y = start_point[1] + x_steps * vector[1]
  

  x = int(x)
  y = int(y)

  return (x,y)
  
def filtrate_trajectory_by_mean_filter(trajectory, filter_size=3):
  """
    Description:
      Filtrate trajectory by mean filter
    Parameters:
      trajectory                    (list(int,int,int))           : trajectory represented as frame,x,y       
      filter_size                   (int)                         : size of mean filter

    Returns:
      filtrated_trajectory         (list(int,int,int))            : filtrated trajectory represented as frame,x,y 
  """
  filtrated_trajectory_x = []
  filtrated_trajectory_y = []
  filtrated_trajectory = []
  for i in range(len(trajectory)):
    filtrated_trajectory_x.append(trajectory[i][1])
    filtrated_trajectory_y.append(trajectory[i][2])
    filtrated_trajectory.append([])
  filtrated_trajectory_x = np.asarray(filtrated_trajectory_x) 
  filtrated_trajectory_y = np.asarray(filtrated_trajectory_y) 
  filtrated_trajectory_x = scipy.ndimage.uniform_filter1d(filtrated_trajectory_x,3)
  filtrated_trajectory_y = scipy.ndimage.uniform_filter1d(filtrated_trajectory_y,3)

  for i in range(len(trajectory)):
    filtrated_trajectory[i] = (trajectory[i][0],filtrated_trajectory_x[i],filtrated_trajectory_y[i])
  return filtrated_trajectory  
############# start code from   http://paulbourke.net/geometry/pointlineplane/######################
def get_distance_point_line(pnt, start, end):
  """
    Description:
      Decide if point is in tolerated L2 distance from line segment.
      Distance calculation from http://paulbourke.net/geometry/pointlineplane/
    Parameters:
      pnt                   (tuple(int,int)): point
      start                 (tuple(int,int)): start point of line
      end                   (tuple(int,int)): end point of line
    Returns:
                            (float)          : distance of point to line
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
  return dist

def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y
  
def length(v):
    x,y, = v
    return math.sqrt(x*x + y*y )
  
def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)
  
def unit(v):
    x,y = v
    mag = length(v)
    if mag == 0:
      return (0,0)
    return (x/mag, y/mag)
  
def distance(p0,p1):
    return length(vector(p0,p1))
  
def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)
  
def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)
############# end code from   http://paulbourke.net/geometry/pointlineplane/######################    
    
def get_center_from_bounding_box(bounding_box):
  """
    Description:
      Get center of bounding box
    Parameters:
      bounding_box        (yolo3.utils.bbox.BoundBox) : bounding box
    Returns:
                            (float,float)             : center of bounding box
  """
  return ((bounding_box.xmin + bounding_box.xmax)/2,(bounding_box.ymin + bounding_box.ymax)/2)
  
def get_bottom_center_from_bounding_box(bounding_box):
  """
    Description:
      Get center of bottom line of bounding box
    Parameters:
      bounding_box        (yolo3.utils.bbox.BoundBox) : bounding box
    Returns:
                            (float,float)             : bottom center of bounding box
  """
  return ((bounding_box.xmin + bounding_box.xmax)/2,bounding_box.ymax )