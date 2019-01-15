import cv2
import numpy as np

filename = '../yale_face/detector_image.png'
path = '/home/lml/desktop/test/ml/yale_face'


def detect(filename):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load(path + '/haarcascade_frontalface_default.xml')

    img = cv2.imread(filename)
    print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print("face", faces)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.namedWindow("vikings detected")
    cv2.imshow("vikings detected", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    detect(filename)