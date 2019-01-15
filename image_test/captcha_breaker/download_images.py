import argparse
import requests
import time
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="downloads", help="path to output directory of images")
ap.add_argument("-n", "--num-images", default=500, help="# of images to download")
args = vars(ap.parse_args())

# url is unloaded
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

for i in range(0, args["num_images"]):
    try:
        r = requests.get(url, timeout=60)
        p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))])
        with open(p, "wb") as f:
            f.write(r.content)
        print("[INFO] downloaded: {}".format(p))
        total += 1
    except Exception as e:
        print("[INFO] error downloading image: {}".format(e))

    time.sleep(0.1)


