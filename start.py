# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

#####boss
from boss_train import Model
#from model_adagrad_update import Model

####emotion
from statistics import mode
from keras.models import load_model
import numpy as np
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
#detection_model_path = 'trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = 'trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]


##image 
emt_img = cv2.imread("./img/emt.png", -1)
# emt_rows, emt_cols, emt_channels = emt_img.shape

# emt_gray = cv2.cvtColor(emt_img, cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)


####boss
model = Model()
model.load()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

# ap.add_argument("-e", "--encodings", required=True,
#     help="path to serialized db of facial encodings")

ap.add_argument("-e", "--encodings", default='encodings.pickle',
    help="path to serialized db of facial encodings")

ap.add_argument("-o", "--output", type=str,
    help="path to output video")

ap.add_argument("-y", "--display", type=int, default=1,
    help="whether or not to display output frame to screen")

ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help="face detection model to use: either `hog` or `hog`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")


#VIDEO_FILE_PATH = './videos/test.mp4'
VIDEO_FILE_PATH = './videos/test2.mp4'
cap = cv2.VideoCapture(VIDEO_FILE_PATH)   # 0 : webcam , video_file_path = 
#vs = VideoStream(src=0).start()


##video save
#재생할 파일의 넓이 얻기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#재생할 파일의 높이 얻기
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#재생할 파일의 프레임 레이트 얻기
fps = cap.get(cv2.CAP_PROP_FPS)

print('width {0}, height {1}, fps {2}'.format(width, height, fps))
#저장할 비디오 코덱
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#저장할 파일 이름
filename = './sprite_with_face_detect3.avi'

out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))

writer = None
time.sleep(2.0)

gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loop over frames from the video file stream
imgNum=0
while True:
    # grab the frame from the threaded video stream
    #frame = vs.read()
    _, frame = cap.read()
    
    
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=420)
    r = frame.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    #encodings = face_recognition.face_encodings(rgb_small_frame, boxes)
   # names = []

    # loop over the facial embeddings
#     for encoding in encodings:
#         # attempt to match each face in the input image to our known
#         # encodings
# #         matches = face_recognition.compare_faces(data["encodings"],
# #             encoding)
#         name = "Unknown"

#         # check to see if we have found a match
# #         if True in matches:
# #             # find the indexes of all matched faces then initialize a
# #             # dictionary to count the total number of times each face
# #             # was matched
# #             matchedIdxs = [i for (i, b) in enumerate(matches) if b]
# #             counts = {}

# #             # loop over the matched indexes and maintain a count for
# #             # each recognized face face
# #             for i in matchedIdxs:
# #                 name = data["names"][i]
# #                 counts[name] = counts.get(name, 0) + 1

# #             # determine the recognized face with the largest number
# #             # of votes (note: in the event of an unlikely tie Python
# #             # will select first entry in the dictionary)
# #             name = max(counts, key=counts.get)

#         # update the list of names
#         names.append(name)
    

    
    
    # loop over the recognized faces
    gender_text=''
    emotion_text=''
    imgNum+=1
    
    print('num face :', len(boxes))
    for (top , right , bottom, left) in boxes:
        # rescale the face coordinates
        rect = [top , right , bottom, left]
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)
        
#         top = top*4
#         right = right*4
#         bottom = bottom*4
#         left = left*4
        y = top -2 if top - 15 > 15 else top + 15

        ##emotion
        x1, x2, y1, y2 = apply_offsets(rect, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]
        
        x1, x2, y1, y2 = apply_offsets(rect, emotion_offsets)
        gray_face = grayframe[y1:y2, x1:x2]
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
            gray_face = preprocess_input(gray_face, False)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion_max = np.max(emotion_classifier.predict(gray_face))
            emotion_text = emotion_labels[emotion_label_arg]
            #emotion_window.append(emotion_text)
            print('^^',emotion_text)


            rgb_face = np.expand_dims(rgb_face, 0)
            rgb_face = preprocess_input(rgb_face, False)
            gender_prediction = gender_classifier.predict(rgb_face)
            gender_max = np.max(gender_classifier.predict(rgb_face))
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]
           # gender_window.append(gender_text)

            cv2.putText(frame, emotion_text +':'+ str(emotion_max), (left, y-15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (100, 0, 255), 2)
            cv2.putText(frame, gender_text +':'+ str(gender_max), (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

        except:

            cv2.putText(frame, emotion_text +':'+ str(emotion_max), (left, y+bottom), cv2.FONT_HERSHEY_SIMPLEX,0.5, (100, 0, 255), 2)
            cv2.putText(frame, gender_text +':'+ str(gender_max), (left, y+15+bottom), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        
        image = frame[top: bottom, left: right]
        result, oy = model.predict(image)
        #cv2.rectangle(frame,( x - int(w / 3),y - int(h / 3)),(x+w+int(w/2),y + h + int(h / 3)),(0,0,255),3)

        if result == 0 :  # boss
            print('Boss is approaching')
#             cv2.rectangle(frame,( left-right//6,top-bottom//6 ),(right+right//6,bottom+bottom//6),(0,255,0),2)
#             cv2.putText(frame, 'Oracle YU : '+str(round(oy,4)), (left-5, top-5), font, 0.8, (153,102,0),2)
            cv2.rectangle(frame, (left, top), (right, bottom),(100, 255, 255), 2)
            cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (100, 255, 255), cv2.FILLED)
            cv2.putText(frame, 'Oracle YU : '+str(round(oy,4)), (left + 6, bottom+13), font,0.5, (0, 0, 0), 1)
        else:
            print('Not boss')
            #cv2.rectangle(frame,( left-right//6,top-bottom//6 ),(right+right//6,bottom+bottom//6),(0,0,255),2)
            cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, '^_^'+str(round(oy,4)), (left + 6, bottom +13), font, 0.5, (0, 0, 0), 1)
        # draw the predicted face name on the image
#         cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
#         cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (0, 0, 255), cv2.FILLED)
#         cv2.putText(frame, '^_^', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        out.write(frame)


    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20,
            (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces t odisk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()
cap.release()
out.release()


# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
