#use this script to go through a video, run inference on each frame, save the predictions to a json file so you can use it for testing
#later without continuosuly runing inference (which is expensive) as well as saving the frames with bounding boxes so you can turn those into a youtube vide if needed
import cv2
import json

from roboflow import Roboflow
rf = Roboflow(api_key="PL4u1GwBJKcV9VpiW16i")
project = rf.workspace().project("york-ats-themepark")
model = project.version(1).model

input_video = '/Users/tobieabel/Desktop/video_frames/v3 demo.mp4'
output_json_folder = '/Users/tobieabel/Desktop/video_frames/json_v4_demo/'
idx = 1
cap=cv2.VideoCapture(input_video)
while cap.isOpened():
    ret, frame = cap.read()
    inference = model.predict(frame, confidence=50, overlap=30)
    predictions = inference.json()

    if len(predictions['predictions']):
        detections = []  # a list that will contain a dictionary of the prediction coordinates with label

        for i in predictions['predictions']:
            x = i['x']
            y = i['y']
            height = i['height']
            width = i['width']
            bounding_box = [x, y, height, width]
            label = i['class']
            confidence = str(i['confidence'])
            confidence = confidence[:4]  # just take first 4 characters of string

            x1 = int(bounding_box[0] - bounding_box[3] / 2)
            x2 = int(bounding_box[0] + bounding_box[3] / 2)
            y1 = int(bounding_box[1] - bounding_box[2] / 2)
            y2 = int(bounding_box[1] + bounding_box[2] / 2)

            detection = [x1, x2, y1, y2, label, confidence]
            detections.append(detection)
        print(*detections, sep="\n")

        # save the json file
        json_file_name = output_json_folder + str(idx) + '.json'
        with open(json_file_name, 'w') as file:
            json.dump(detections, file)

        for i in detections:
            cv2.rectangle(frame,(i[0],i[2]),(i[1],i[3]),(255,255,255),2)
            cv2.putText(frame, i[4],(i[0],i[2] -5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0))

        cv2.imwrite('/Users/tobieabel/Desktop/video_frames/v4_demo_with_bbox/' + str(idx) + '.jpeg',frame)
        idx += 1
