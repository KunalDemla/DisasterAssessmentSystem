import numpy as np
from PIL import Image

img = Image.open(r'C:/Users/Ashima/OneDrive/Desktop/SuperReso_Python/Images/d.png')

# img.show()
from ISR.models import RDN, RRDN

# model = RDN(weights='noise-cancel')
model = RRDN(weights='gans')

img.resize(size=(img.size[0]*4, img.size[1]*4), resample=Image.BICUBIC)

sr_img = model.predict(np.array(img))
new = Image.fromarray(sr_img)

new.save(r'C:/Users/Ashima/OneDrive/Desktop/SuperReso_Python/Images/dnew.png') 
