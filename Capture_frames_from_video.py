import cv2
#video1 folder should be the front of the house, video 2 should be the back garden
video_path = '/Users/tobieabel/Desktop/video1.mp4'

cap = cv2.VideoCapture(video_path)
idx = 1
frame_index = 1
frame_skip = 1
while cap.isOpened() and frame_index < 20000:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cv2.imwrite('/Users/tobieabel/Desktop/video_frames/video1/' + str(idx) + '.jpeg',frame)
    idx += 1
    print(frame_index)
    frame_index += frame_skip

