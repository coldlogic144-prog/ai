import numpy as np
from PIL import Image
img=Image.open("number.jpeg")
img=img.resize((300,300))
img=img.convert("L")
pixals=np.array(img)
pixals=pixals/255.0
print(pixals.shape)
pixals_1d=pixals.flatten()
print(pixals_1d.shape)
