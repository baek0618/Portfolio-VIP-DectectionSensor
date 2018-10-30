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
import glob 
from PIL import Image

img_size = 64
img_path = './crop_img/'


# load the known faces and embeddings
print("[INFO] img_path :",img_path)

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] img_size :", img_size)

#vs = VideoStream(src=0).start()

VIDEO_FILE_PATH = './videos/test_v.mp4'
cap = cv2.VideoCapture(VIDEO_FILE_PATH)

writer = None
time.sleep(2.0)
imgNum = 0
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    _, frame = cap.read()
    if frame is None:
        break;
    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,
        model='cnn')
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    
    # loop over the recognized faces
    for top, right, bottom, left in boxes:
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        #cv2.rectangle(frame, (left, top), (right, bottom),
        #    (0, 255, 0), 2)
        #y = top - 15 if top - 15 > 15 else top + 15
       # cv2.putText(frame, 'zzzz', (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        #    0.75, (0, 255, 0), 2)

        #if imgNum % 10 == 0:
        cropped = frame[top: bottom, left: right]
        cv2.imwrite(img_path+VIDEO_FILE_PATH.replace('.mp4','')[-4:-1] + str(imgNum) + ".png", cropped)
        imgNum+=1


    # check to see if we are supposed to display the output frame to
    # the screen

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

###img resize 
imglist =glob.glob(img_path+"*png")
print(imglist)
print(img_path+"*png")
img = Image.open(imglist[0])

for img_path in imglist:
    img = Image.open(img_path)
    img.resize((img_size,img_size)).save(img_path)

# do a bit of cleanup
cv2.destroyAllWindows()


# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
