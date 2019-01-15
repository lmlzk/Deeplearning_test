from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

cas_path = "/home/lml/desktop/test/ml/yale_face/"
cas_name = "haarcascade_frontalface_default.xml"

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", default=cas_name,
                help="path to where the face cascade resides")
ap.add_argument("-m", "--model", default="../output/smile/minivggnet.hdf5", help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video", default="../../yale_face/test_video1.mp4", help="path to the (optional) video file")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
detector.load("/home/lml/desktop/test/ml/yale_face/haarcascade_frontalface_default.xml")
print(detector)
model = load_model(args["model"])

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])
    print("camera")

while True:
    grabbed, frame = camera.read()
    # print(grabbed, frame)

    if args.get("video") and not grabbed:
        break

    # frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    # rects = detector.detectMultiScale(gray, 1.3, 5)
    # print("faces", rects)
    for fX, fY, fW, fH in rects:
        roi = gray[fY: fY+fH, fX: fX+fW]
        roi = cv2.resize(roi, (128, 97))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        notSmiling, smiling = model.predict(roi)[0]
        print(notSmiling, smiling)
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        cv2.putText(frameClone, label, (fX, fY-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (255, 0, 0), 2)

    cv2.imshow("Face", frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

