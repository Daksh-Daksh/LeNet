from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def build_model(out_dim):

    model = Sequential()
    # Create CN layer 1
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(64, 64, 1), activation='relu'))
    # Create Max-Pool 1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Create CN layer 2
    model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', input_shape=(64, 64, 1), activation='relu'))
    # Create Max-Pool 2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add Dropout layer
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Fully connected
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))
    model.summary()
    print("")

    return model
