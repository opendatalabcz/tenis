import os
import sys 
from court_detection import get_specific_court_points_and_homographies, get_specific_court_points_and_homographies_parallel
from utils import save_frames_in_time_interval_from_video,load_frames_in_time_interval_from_video, linear_interpolation_of_trajectory, filtrate_trajectory_by_mean_filter, save_images_as_video
from events_detection import get_events_sequence
from visualization import visualize
from detect_players import get_players_bounding_boxes_in_images, get_yolov3_model, yolo_predict
from joblib import dump, load
from tracknet import get_tracknet_model, get_detections_in_images
from TLDA import run_TLDA
import gc
import numpy as np
from joblib import dump


#path to file with start and end frames for sequences to be processed
path_to_clips_frames = '.\\V008_annotation_clips.csv'
#path to YOLOv3 implementation configuration file, path to weights specified there
path_to_yolo_config = '.\\yolo3\\config.json'
#path to TrackNet weights
path_to_TrackNet_weights = '.\\Tracknet\\Code_Python3\\TrackNet_Three_Frames_Input\\weights\\model.3'
# path to video
path_to_video = '.\\' + 'V008' + '.mp4'
#path to directory to save results
path_to_dir_to_save_results = '.\\results\\'

#should parallel cpu computation be used
parallel_cpu = True
#should be video visualized to file
visualize_in_video = True

#size of artificial parallel court model, doesnt matter for running,  could be useful for more complex visualizations
len_of_base_line_in_homography = 200
margin_in_homography = 10

#if parallelelization on CPU should be used
if parallel_cpu == True:
  from multiprocessing import Process, Pipe, cpu_count, current_process, Manager,Pool

#for version of FW algorithm change variable in TLDA.py


serve_categories = ['SFL','SFR','SNL','SNR']


def write_results(path_to_dir_to_save_results, events_indexes_classificated, speed_of_serve, frame_start, frame_end):
  f = open(path_to_dir_to_save_results + 'events_' + str(frame_start) + '_'+ str(frame_end) + '.csv', "w")
  for i in range(len(events_indexes_classificated)):
    if events_indexes_classificated[i][1] in serve_categories:
        f.write(f"{events_indexes_classificated[i][0] + frame_start},{events_indexes_classificated[i][1]},{speed_of_serve}\n")
    else:
      f.write(f"{events_indexes_classificated[i][0] + frame_start},{events_indexes_classificated[i][1]}\n")
  f.close()
  
def main():  


  isExist =  os.path.exists(path_to_video)
  isExits3 = os.path.exists(path_to_clips_frames)
  
  if isExits3 == False:
    print(f"path_to_clips_frames does not exist: {path_to_clips_frames} ")
    return
  with open(path_to_clips_frames) as f:
  
    #create pool of processes
    if parallel_cpu == True:
      num_cores = cpu_count()
      pool = Pool(processes=num_cores)
      
    cnt = 0  
    #load CNN models
    model_yolov3 = get_yolov3_model(path_to_yolo_config)
    model_TrackNet = get_tracknet_model(path_to_TrackNet_weights)  
    #read file with start frame and end frame number of frame sequence to be processed 
    while True:
      if cnt == 0:
        line = f.readline()#skip header
        cnt = cnt + 1
        continue
      else:
        line = f.readline()
        if line == '':
          break
        words = line.split(',')

        frame_number1 = int(words[0])
        frame_number2 = int(words[1])
        path_to_save_video = path_to_dir_to_save_results  + str(frame_number1)+'_'+str(frame_number2)+'.avi'
        print(f"frame_number1 {frame_number1} frame_number2 {frame_number2} path_to_video {path_to_video}")
        
        if isExist == True:
          isExist2 =  os.path.exists(path_to_dir_to_save_results)
          if isExist2 != True:
            os.makedirs(path_to_dir_to_save_results)
          #load frames
          images,fps = load_frames_in_time_interval_from_video(path_to_video, frame_number1, frame_number2)
          #control if frames loaded correctly
          if images is None:
            print(f"sequence {frame_number1},{frame_number2}:loading frames error, frames loaded 0, should be loaded {frame_number2 - frame_number1 + 1}")
            continue
          if len(images) != (frame_number2 - frame_number1 + 1):
            print(f"sequence {frame_number1},{frame_number2}:loading frames error, frames loaded {len(images)}, should be loaded {frame_number2 - frame_number1 + 1}")
            continue
            
          image_height, image_width = images[0].shape[0:2]
          
          #find court
          if parallel_cpu == False:
            corner_points_per_image, court_found_per_image, homographies_image_to_artificial_court_per_frame, homographies_artificial_court_to_image_per_frame = get_specific_court_points_and_homographies(images, len_of_base_line_in_homography, margin_in_homography, image_height, image_width)
          else:
            corner_points_per_image, court_found_per_image, homographies_image_to_artificial_court_per_frame, homographies_artificial_court_to_image_per_frame_parallel = get_specific_court_points_and_homographies_parallel(images, len_of_base_line_in_homography, margin_in_homography, image_height, image_width, num_cores, pool)
          #detect players by YOLOv3
          players_detections = get_players_bounding_boxes_in_images(model_yolov3, images, corner_points_per_image,path_to_yolo_config)
          #detect ball by TrackNet
          detections_in_frames = get_detections_in_images(images, model_TrackNet)
          #find trajectory by TLDA
          trajectory = run_TLDA(detections_in_frames,corner_points_per_image,image_width,image_height)
          #find events
          if trajectory is None:
            print(f"{frame_number1},{frame_number2}: trajectory not found")
            events_indexes_classificated = [[0,'trajectory not found for sequence ' + str(frame_number1) + '_' + str(frame_number2)]]
            speed_of_serve = 'x'
          else:
            trajectory = linear_interpolation_of_trajectory(trajectory)
            filtrated_trajectory = filtrate_trajectory_by_mean_filter(trajectory)
            events_indexes_classificated, tenis_exchange_ending, speed_of_serve = get_events_sequence(trajectory, filtrated_trajectory, players_detections, corner_points_per_image, homographies_image_to_artificial_court_per_frame, image_width, image_height, len_of_base_line_in_homography, margin_in_homography, fps)
            #visualize video
            if visualize_in_video == True:
              images = visualize(images, events_indexes_classificated, speed_of_serve, trajectory, corner_points_per_image)
              save_images_as_video(images, path_to_save_video,fps)    
            
          images = []
          #write results to file
          write_results(path_to_dir_to_save_results, events_indexes_classificated, speed_of_serve, frame_number1, frame_number2)
        else:
          print("path to video does not exist") 


if __name__ == "__main__":
  main()