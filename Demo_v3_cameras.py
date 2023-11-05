import numpy as np
import cv2
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
import time
from threading import Thread
import sys #used to print out memory size of variable for debug purposes

Stop = False
def stop_flag():
    global Stop
    #Could not get the Stop flag to effectively stop the loop inside the thread (create frames method) - probqbly something to do with Streamlit and threading not working well together
    if st.session_state.flag == 'true':
        Stop = True
    print("Value of Stop at top of page is " + str(Stop))

if 'flag' in st.session_state:
    stop_flag()


#initialise cameras
def Initialise_Video(): #Remember to remove credentials if sending this to github
    global camera1
    global camera2
    camera1 = cv2.VideoCapture()
    camera1.open("rtsp://admin:h*201@192.168.1.66:554/Streaming/Channels/2")#indoor camera
    camera2 = cv2.VideoCapture()
    camera2.open("rtsp://admin:h@201@192.168.1.65:554/Streaming/Channels/2")#garden camera
    time.sleep(1)



COLORS = sv.ColorPalette.default()

#Define entry and exit areas on image (got the cordinates by drawing zones using https://blog.roboflow.com/polygonzone/)
#Zone_in is garden bottom half and front of house bottom half - red colour
ZONE_IN_POLYGONS = [
    np.array([[640, 154],[0, 242],[0, 360],[640, 360]]),
    np.array([[650, 162],[986, 158],[990, 360],[646, 360]]),
]
#Zone_out is garden top half and front of house top half - green colour
ZONE_OUT_POLYGONS = [
    np.array([[642, 0],[978, 0],[982, 142],[654, 146]]),
    np.array([[0, 0],[634, 0],[638, 146],[2, 222]]),
]
def initiate_poylgon_zones(polygons:list[np.ndarray],frame_resolution_wh:tuple[int,int],triggering_position:sv.Position=sv.Position.CENTER)->list[sv.PolygonZone]:
    return[sv.PolygonZone(polygon,frame_resolution_wh,triggering_position)for polygon in polygons]

class DetectionsManager: #this is the rules engine
    def __init__(self, total_count: int) -> None:
        self.tracker_id_to_zone_id: Dict[int, str] = {}
        self.total_count = total_count


    #update function takes the list of detections triggered by a zone and maps the tracker ID to either in or out
    def update(self,detections: sv.detection, detections_zone_in: list[sv.detection], detections_zone_out: list[sv.detection]) -> sv.detection:
        for detection in detections_zone_in:
            #print('Zone in detection  ', detection)
            if np.any(detection.tracker_id):#this tests if there are any tracker id's.  If not the for loop below crashes
                for tracker_id in detection.tracker_id:
                    if tracker_id in self.tracker_id_to_zone_id:
                        #print(self.tracker_id_to_zone_id[tracker_id])
                        if self.tracker_id_to_zone_id[tracker_id] == 'out':#if current value is out then this detection has crossed zones
                            self.total_count += 1 #add one to the count as an 'out' has become an 'in'
                            self.tracker_id_to_zone_id[tracker_id] = 'in' # and update zone in dictionary to reflect this
                    else:
                        self.tracker_id_to_zone_id[tracker_id] = 'in' #this means tracker ID is new so add to the dictionary

        for detection in detections_zone_out:
            #print('Zone out detections  ', detection)
            if np.any(detection.tracker_id): #this tests if there are any tracker id's.  If not the for loop below crashes
                for tracker_id in detection.tracker_id:
                    if tracker_id in self.tracker_id_to_zone_id:
                        #print(self.tracker_id_to_zone_id[tracker_id])
                        if self.tracker_id_to_zone_id[tracker_id] == 'in':#if current value is in then this detection has crossed zones
                            self.total_count -= 1 #minus one to the count as an 'in' has become an 'out'
                            self.tracker_id_to_zone_id[tracker_id] = 'out' # and update zone in dictionary to reflect this
                    else:
                        self.tracker_id_to_zone_id[tracker_id] = 'out' #this means tracker ID is new so add to the dictionary

        #Need new statement which filters the detections so it only shows those from within a zone - although not sure that matters for this use case as zones cover whole field of view
        #detections.class_id = np.vectorize(lambda x: self.tracker_id_to_zone_id.get(x, -1))(detections.tracker_id)#i don't understand what this is doing so need to come back to it


        return self.total_count

class VideoProcessor: #this is the main class which controls the process flow
    def __init__(self, source_weights_path: str, target_video_path: str = None,
        confidence_threshold: float = 0.4, iou_threshold: float = 0.7, total_count: int = 0) -> None:
        self.source_weights_path = source_weights_path
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_video_path = target_video_path
        self.total_count = total_count

        self.model = YOLO(self.source_weights_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2)
        self.video_info = sv.VideoInfo(1280,360,25)#replacing from_video_path method with hardcoded width, height and FPS for now

        self.zone_in = initiate_poylgon_zones(ZONE_IN_POLYGONS,self.video_info.resolution_wh,sv.Position.CENTER)
        self.zone_out = initiate_poylgon_zones(ZONE_OUT_POLYGONS,self.video_info.resolution_wh,sv.Position.CENTER)
        self.detections_manager = DetectionsManager(self.total_count)

        self.thread1 = Thread(target = self.Create_frames, args=())
        self.thread1.daemon = True
        self.thread1.start()


    def Create_frames(self):  # this function runs in a separate thread initiated in the main class constructor
        IDX = 0
        global Stop
        while Stop == False:
            ret1, frame1 = camera1.read()  # indoor camera
            ret2, frame2 = camera2.read()  # garden camera
            if ret1 is False or ret2 is False:  # test if frame cannot be captured, if so link is probably down so reinitialise streams
                IDX += 1
                print("no frame returned ",IDX, camera1.getExceptionMode())#print out some videocapture() parameters like status
                if IDX % 10 == 0:
                    Initialise_Video()

            else:
                self.cam1and2 = np.concatenate((frame1, frame2), axis=1) #Concatenate the two frames, axis 1 displays horizontally

                IDX += 1#print IDX to see rate of frames captured and concatenated in this thread
                if IDX % 100 == 0:
                    print(IDX, "thread 1  ", sys.getsizeof(self.cam1and2))
                    print("Value of Stop inside the thread is " + str(Stop))
                # store cam1and2 image in a buffer list that main loop can access

    def process_video(self):
        IDX = 0
        while True:
            t1 = cv2.getTickCount()#start FPS calculation
            if self.target_video_path:#Needs rewritting as not using frame_generator.  if you have set video path in call to VideoProcessor class then we will write frames to a video
                with sv.VideoSink(self.target_video_path, self.video_info) as f:
                    for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                        processed_frame = self.process_frame(self.frame)
                        f.write_frame(processed_frame)

            else:#if no video path given then we show the frames live
                #for frame in frame_generator:#may need this if I move frames to a list
                processed_frame = self.process_frame(self.cam1and2)
                t2 = cv2.getTickCount()
                ticks_taken = (t2-t1)/cv2.getTickFrequency()
                FPS = 1/ticks_taken
                cv2.putText(processed_frame,'FPS: {0:.2f}'.format(FPS),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                #cv2.imshow("Live Stream People Counter", processed_frame)#I think this causes the app to crash after 13500 loops due to excessive memory consumption
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)#change the frame to RGB for Streamlit
                frame_placeholder.image(processed_frame, channels="RGB",use_column_width=True)#display in the streamlit app
                IDX += 1#print IDX to see rate of frames processed by model
                if IDX % 100 ==0:
                    print(IDX, "Main thread ", sys.getsizeof(self.process_frame(self.cam1and2)))

                    #cv2.imwrite('/Users/tobieabel/PycharmProjects/demo/Static/Processed_Frame.jpeg', processed_frame)#write frame to local directory for display
                if cv2.waitKey(1) & 0xFF ==ord("q"):
                    break

            cv2.destroyAllWindows()

    def process_frame(self,frame: np.ndarray)-> np.ndarray:
        #consider resizing the frame tp 180x640 for both training and inference to see of this speeds things up
        result = self.model(frame, device = "mps", verbose = False, conf=self.conf_threshold,iou=self.iou_threshold)[0]#add the device parameter to run this on the Mac's GPU which significantly speeds up inference.  See Ultralytics docs for device type for each processor type
        detections = sv.Detections.from_ultralytics(result)#pass the YOLO8 inference results through supervision to use their detections object which is easier to process
        detections = detections[detections.class_id == 0]#filter the list of detections so it only shows category '0' which is people
        detections = self.tracker.update_with_detections(detections)#pass the detections through the tracker to add tracker ID as additional field to detections object

        #filter out detections not triggered within a zone and add the detections to lists for zone in and zone out
        detections_zone_in = []
        detections_zone_out = []
        for zone_in, zone_out in zip(self.zone_in,self.zone_out):
            detection_zone_in = detections[zone_in.trigger(detections)]#this is an Supervision function to test if a detection occured within a zone
            detections_zone_in.append(detection_zone_in)
            detection_zone_out = detections[zone_out.trigger(detections)]#this is an Supervision function to test if a detection occured within a zone
            detections_zone_out.append(detection_zone_out)


        total_count = self.detections_manager.update(detections,detections_zone_in,detections_zone_out)#call to the detections manager class 'rules engine' for working out which zone a detection was triggered in

        return self.annotate_frame(frame,detections,total_count)

    def annotate_frame(self,frame: np.ndarray, detections: sv.Detections,total_count:int)-> np.ndarray:
        annotated_frame = frame.copy()
        for i,(zone_in,zone_out) in enumerate(zip(self.zone_in,self.zone_out)):#use enumerate so you get the index [i] automatically
            annotated_frame = sv.draw_polygon(annotated_frame,zone_in.polygon,COLORS.colors[0])#draw zone in polygons
            annotated_frame = sv.draw_polygon(annotated_frame,zone_out.polygon,COLORS.colors[1])#draw zone out polygons

        if detections:#need to check some detections are found before adding annotations, otherwise list comprehension below breaks
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]#list comprehension to return list of tracker_ID's to use in label
            annotated_frame = self.box_annotator.annotate(annotated_frame,detections,labels=labels)
            annotated_frame = self.trace_annotator.annotate(annotated_frame,detections)

        annotated_frame = sv.draw_text(scene=annotated_frame, text="Count of People Currently In", text_anchor=sv.Point(x=1130, y=150), text_scale=0.6, text_thickness=1,background_color=COLORS.colors[0])
        annotated_frame = sv.draw_text(scene=annotated_frame,text=str(total_count),text_anchor=sv.Point(x=1118, y=226),text_scale=2,text_thickness=5,background_color=COLORS.colors[0],text_padding=40)

        return annotated_frame



def start(Reset_Count = 0):
    if 'flag' not in st.session_state:
        st.session_state.flag = 'false'
    st.session_state.flag = 'false'
    Initialise_Video()
    global Processor
    Processor = VideoProcessor(
       source_weights_path='yolov8nPeopleCounterV2.pt', total_count=int(Reset_Count),
        #target_video_path='/Users/tobieabel/Desktop/video_frames/Youtube/v3_a demo_annotated.mp4',
        )
    time.sleep(0.5)
    Processor.process_video()

def stop():#Cannot find a way to kill the create frames thread with Streamlit
    if 'flag' not in st.session_state:
        st.session_state.flag = 'true'
    st.session_state.flag = 'true'

#front end using streamlit - run in terminal with command: streamlit run /Users/tobieabel/PycharmProjects/demo/Demo_v3_cameras.py

frame_placeholder = st.image("Team Pic.jpg")#placeholder for video stream


start_people_counter = st.button(''':green[Start People Counter]''')
stop_button_pressed = st.button(":red[Stop]")
Reset_Count = st.text_input('How many people are currently inside?',value = 0)


if start_people_counter:
    if Reset_Count.isdigit():
        start(int(Reset_Count))
    else:
        st.write("Please only enter numbers in the input box")

if stop_button_pressed:
    stop()




