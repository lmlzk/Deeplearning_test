from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from image_test.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from image_test.preprocessing.simplepreprocessor import SimplePreprocessor
from image_test.datasets.simpledatasetsloader import SimpleDatasetLoader
from image_test.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="../flowers", help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading image...")
imagePath = list(paths.list_images(args["dataset"]))

x = y = 128
sp = SimplePreprocessor(x, y)
iap = ImageToArrayPreprocessor()

sd1 = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sd1.load(imagePath, verbose=500)
data = data.astype("float") / 255.0

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)
target_name = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
classes = len(target_name)
model = ShallowNet.build(width=x, height=y, depth=3, classes=classes)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=100, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=target_name))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

