from PIL import Image
import numpy as np
import glob
import math

m0=124.92
print("Remember to fix mean variable to the correct value")
print("Mean value used for calculating stddev: (c0)=({})".format(m0))

mc0 = 0.0
n = 0.0
for file in glob.glob("./640/train/**/*.png", recursive=True):
    img = Image.open(file)
    dat = np.array(img)
    pixels = dat.shape[0]*dat.shape[1]
    c0 = np.sum(np.square(dat[:,:] - m0))/pixels
    mc0 += c0
    n += 1.0

mc0 /= n

mc0 = math.sqrt(mc0)

print("c0={}".format(mc0))
