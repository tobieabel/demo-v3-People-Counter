import numpy as np
import cv2
import os

input_folder1 = '/Users/tobieabel/Desktop/video_frames/video1/'
input_folder2 = '/Users/tobieabel/Desktop/video_frames/video2/'
output_folder = '/Users/tobieabel/Desktop/video_frames/ConcatVideo/'

#create list of files from each input folder directory
video1_files = os.listdir(input_folder1)
video1_files.remove('.DS_Store')
video2_files = os.listdir(input_folder2)
video2_files.remove('.DS_Store')
print(len(video1_files), " ", len(video2_files))

#sort the file lists chronologically so they are both in the same sequence
sorted_video1_files = sorted(video1_files, key=lambda x:int(x.split('.')[0]))
sorted_video2_files = sorted(video2_files, key=lambda x:int(x.split('.')[0]))

#take a file from each list and concatenate them
for x,y in (zip(video1_files,video2_files)):
    x_image_path = os.path.join(input_folder1, x)
    x1 = cv2.imread(x_image_path)
    y_image_path = os.path.join(input_folder2, y)
    y1 = cv2.imread(y_image_path)
    cam1and2 = np.concatenate((x1,y1), axis=1) #axis 1 displays horizontally
    cv2.imwrite(output_folder + str(x),cam1and2)


