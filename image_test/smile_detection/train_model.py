from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from image_test.nn.conv.lenet import LeNet
from image_test.nn.conv.minivggnet import MiniVGGNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os
import imageio
# http://cvc.cs.yale.edu/cvc/projects/yalefaces/yalefaces.html

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="../../yale_face", help="path to input dataset of faces")
ap.add_argument("-m", "--model", default="../output/smile/minivggnet.hdf5", help="path to output model")
args = vars(ap.parse_args())

epochs = 100
data = []
labels = []
imagelist = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")
image_list = sorted(list(paths.list_files(args["dataset"], validExts=imagelist)))
for imagePath in image_list:
    image = cv2.imread(imagePath, flags=cv2.IMREAD_GRAYSCALE)
    if image is None:
        image = imageio.imread(imagePath)
        # print(image)
    # cv2.IMREAD_GRAYSCALE
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(dir(image))
    # print(image.shape, imagePath)
    image = imutils.resize(image, width=128)
    image = img_to_array(image)
    data.append(image)

    imagename = imagePath.split(os.path.sep)[-1]
    label = imagename.split(".")[0].split("_")[-1]

    label = "smiling" if label in ["happy", "wink"] else "not_smiling"
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

le = LabelBinarizer().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

trainX, testX, trainY, testY = train_test_split(data,
        labels, test_size=0.50, stratify=labels, random_state=42)

print("[INFO] compiling model...")
x, y = 128, 97
# model = LeNet.build(width=x, height=y, depth=1, classes=2)
model = MiniVGGNet.build(width=x, height=y, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              class_weight=classWeight, batch_size=64, epochs=epochs, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=le.classes_))

print("[INFO] serializing network...")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Smiling Training")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
# plt.savefig()
plt.show()