from image_test.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from image_test.preprocessing.simplepreprocessor import SimplePreprocessor
from image_test.datasets.simpledatasetsloader import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="../flowers", help="path to input dataset")
ap.add_argument("-m", "--model", default="./output/shallownet_weights.hdf5",
                help="path to pre-trained model")
args = vars(ap.parse_args())

target_name = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
classes = len(target_name)

# randomly sample
print("[INFO] sampling image...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

x = y = 128
sp = SimplePreprocessor(x, y)
iap = ImageToArrayPreprocessor()

sd1 = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sd1.load(imagePaths)
data = data.astype("float") / 255.0

print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

for i, imagePath in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    print("[INFO] imagePath: {}".format(imagePath))
    name = imagePath.split("/")[-2]
    cv2.putText(image, "Label: {}".format(target_name[preds[i]]),
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, "Truth: {}".format(name),
                (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
