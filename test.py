import keras
import numpy as np
import glob
from keras.preprocessing import image

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# dictionary -> { key: value }
test_label_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5}

file_path = 'DATA/'
f_names = glob.glob(file_path + '*.png')
img = []

model = keras.models.load_model('HDF5_file.h5')

for i in range(len(f_names)):
    # load every figures in the "file_path" you set
    images = image.load_img(f_names[i], target_size=(64,64),color_mode = "grayscale")
    # translate figure to array that we need for calculating during predicting
    x = image.img_to_array(images)
    # let x be the same size we need for model
    x = np.expand_dims(x, 0)
    print('loading no.%s image' % i)

    y = model.predict(x)
    print(y)