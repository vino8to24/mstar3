import numpy as np
from PIL import Image

img_num = 697
image_size = 128*128
label_size = 1
record_size = image_size + label_size
f = open('mstar3_train.bin')
data = np.fromfile(f, np.float32)
record = np.zeros(shape=(img_num, record_size))
images = np.zeros(shape=(img_num, image_size))
labels = np.zeros(shape=(img_num, label_size))
image = np.zeros(shape=(128, 128))
for i in range(img_num):
    record[i, :] = data[record_size*i:record_size*(i+1)]
for i in range(img_num):
    labels[i, :] = record[i, 0]
    images[i, :] = record[i, 1:image_size+1]
    if i%50 == 0:
        image = np.reshape(images[i, :], newshape=(128, 128))
        print(image*255)
        image = Image.fromarray(image*255)
        image.show()



