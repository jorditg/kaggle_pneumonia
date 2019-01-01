from PIL import Image
import numpy as np
import glob

mc0=0.0
n = 0.0
for file in glob.glob("./640/train/**/*.png", recursive=True):
    img = Image.open(file)
    dat = np.array(img)
    c0 = dat[:,:].mean()
    mc0 += c0
    n += 1.0

mc0 /= n

print("c0={}".format(mc0))
