import os
import numpy as np
from PIL import Image
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import random
import matplotlib.pyplot as plt
from time import *
from LeNet import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

label_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}


def return_label(folder_name):
    return label_dict[folder_name]


def Draw_Model_Figure(History, Model_HDF5_name):
    """
        print(History.history.keys()) -> dict_keys(['acc', 'val_acc', 'val_loss', 'loss', 'lr'])
    """
    plt.figure()  # initialize. Without this, when using Loop will draw everything on same image
    plt.subplot(2, 1, 1)
    plt.plot(History.history['acc'])
    plt.plot(History.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')

    plt.subplots_adjust(wspace=0, hspace=1)

    plt.subplot(2, 1, 2)
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')

    Model_name = Model_HDF5_name[:-3]
    Output_Image_name = Model_name + '.png'
    plt.savefig(Output_Image_name)
    plt.close()
    # plt.show()


def get_training_set(train_filepath):
    training_tmp = os.listdir(train_filepath)
    training_set = []
    for i in training_tmp:
        tmp = os.path.join(train_filepath, i)
        if os.path.isdir(tmp):
            # print(i, return_label(i))
            training_set.append(tmp)

    Train_data_list = []
    Train_label_list = []
    for _, train_subpath in enumerate(training_set):
        tmp = os.listdir(train_subpath)
        label = return_label(train_subpath.split('\\')[-1])

        for _, img in enumerate(tmp):
            img_filepath = os.path.join(train_subpath, img)

            if os.path.splitext(img)[-1] != '.png':
                print('[Invalid data type] ', img_filepath)
                continue

            img_tmp = Image.open(img_filepath)  # input image
            # print(img_tmp.format, img_tmp.size, img_tmp.mode)
            # img_tmp.show()
            img_num_arr = np.array(img_tmp).reshape(64, 64, 1)
            img_num_arr = img_num_arr.astype('float32')
            img_num_arr /= 255

            Train_data_list.append(img_num_arr)
            Train_label_list.append(to_categorical(label, out_dim))

    c = list(zip(Train_data_list, Train_label_list))
    random.shuffle(c)
    x, y = zip(*c)
    validation_len = int(len(x) * 0.85)
    x_train_arr = np.array(x[:validation_len])
    x_valid_arr = np.array(x[validation_len:])
    y_train_arr = np.array(y[:validation_len])
    y_valid_arr = np.array(y[validation_len:])

    return x_train_arr, y_train_arr, x_valid_arr, y_valid_arr


if __name__ == '__main__':
    out_dim = 6  # Total 6 faces in dice

    Model_Train_time = 1
    for train_loop in range(0, Model_Train_time):
        train_filepath = 'C:\Users\IOPLAB\PycharmProjects\dice 1'
        x_train, y_train, x_valid, y_valid = get_training_set(train_filepath)

        model = build_model(out_dim)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        train_history = model.fit(x_train, y_train, validation_split=0.2, epochs=80, batch_size=128,
                                  verbose=1)
        Now_time = localtime()  # return format timestamp
        Model_Name_Now_Time = '0' + str(Now_time[1]) + str(Now_time[2]) + '_' + str(Now_time[3]) + \
                              str(Now_time[4]) + '_' + str(Now_time[5])  # avoid duplicated HDF5 file name
        HDF5_file = 'Model_' + Model_Name_Now_Time + '.h5'
        model.save(HDF5_file + '.h5')
        print('Model - ', HDF5_file)
        print('Start')
        c = ModelCheckpoint(HDF5_file, monitor='val_acc', verbose=0, period=1)
        History = model.fit(x_train, y_train,
                            batch_size=128,
                            epochs=80,
                            validation_data=(x_valid, y_valid),
                            callbacks=[c],
                            verbose=1
                            )
        print('End')
        Draw_Model_Figure(History, HDF5_file)

        K.clear_session()
        print()
