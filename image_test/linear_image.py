import numpy as np
import cv2

labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
np.random.seed(1)

x = y = 32

W = np.random.randn(5, x*y*3)
b = np.random.randn(5)

path = "../flowers/daisy/004daisy.jpg"
orig = cv2.imread(path)
images = cv2.resize(orig, (x, y)).flatten()

scores = W.dot(images) + b


for label, score in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Image", orig)
cv2.waitKey(0)