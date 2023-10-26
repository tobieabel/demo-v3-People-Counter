import cv2
import os

# Input folder containing the saved images
image_folder = '/Users/tobieabel/Desktop/video_frames/ConcatVideo/'

# Output video file path
output_video_path = '/Users/tobieabel/Desktop/video_frames/Youtube/v3_a demo.mp4'

# Get the list of image files in the input folder
image_files = os.listdir(image_folder)
image_files.remove('.DS_Store')
def file_sort_key(filename):
    # Extract the numeric portion of the filename
    number = int(os.path.splitext(filename)[0])

    return number

# Sort the files chronologically
sorted_files = sorted(image_files, key=file_sort_key)
print(sorted_files)
# Get the dimensions of the first image to initialize the video writer
first_image_path = os.path.join(image_folder, sorted_files[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape #need to be careful of this.  I scrapped a video from youtube whose resolution was an odd width 1740, height 988
#and these dimensions didn't work with cv2.VideoWriter so I had to use cv2.resize to change the images to 1920x1080 which is the closest accepatable format

# Define the codec and create the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30  # Adjust as needed
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Loop through the image files and write them to the video
for i in sorted_files:
    image_path = os.path.join(image_folder, i)
    image = cv2.imread(image_path)

    # Write the image to the video writer
    video_writer.write(image)

# Release the video writer
video_writer.release()

print(f"Video saved to: {output_video_path}")