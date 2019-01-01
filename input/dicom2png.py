import os
import pydicom
import random
import numpy as np
import glob
import cv2
import pandas as pd

df = pd.read_csv("stage_1_detailed_class_info.csv")

out_size = 1024

val_proportion = 0.10

def convert_dicom2png_and_resize(file, out_size=640):
    dicom = pydicom.dcmread(file)
    img = dicom.pixel_array
    res = cv2.resize(img, dsize=(out_size, out_size), interpolation=cv2.INTER_LANCZOS4)
    return res

files = "./train_dicom/*.dcm"
for file in glob.glob(files):
    id = os.path.splitext(os.path.basename(file))[0]
    res = convert_dicom2png_and_resize(file, out_size)
    classval = df.loc[df['patientId'] == id]
    print(classval)
    classval = classval.values[0][1]
    print(classval)
    if classval == "Normal":
        classdir = "normal"
    elif classval == "Lung Opacity":
        classdir = "opacity"
    else:
        classdir = "not_normal"
    dest = "val" if random.random() < val_proportion else "train"
    print(dest)
    newfile = "./{}/train/{}/{}/{}.png".format(str(out_size), dest, classdir, id)
    print(newfile)
    print(res.shape)
    cv2.imwrite(newfile, res)

files = "./test_dicom/*.dcm"
for file in glob.glob(files):
    id = os.path.splitext(os.path.basename(file))[0]
    res = convert_dicom2png_and_resize(file, out_size)
    newfile = "./{}/test/{}.png".format(str(out_size), id)
    print(newfile)
    cv2.imwrite(newfile, res)
