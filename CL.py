import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, InputLayer, GlobalAveragePooling2D, BatchNormalization, ReLU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

data = np.load('cifar10_ipc50_relax_result_seed1.npy', allow_pickle=True).item()
images = data['condensed_dataset']['x']
labels = data['condensed_dataset']['y']

data_train = np.asarray(images)
labels_train = np.asarray(labels)

# plt.imshow(data_train[0])
# plt.axis('off')
# plt.savefig('GTSRB(size2150clean).png', bbox_inches='tight', pad_inches=0)
# plt.show()

num_subsets = 1
subsets = np.array_split(data_train, num_subsets)
label_subsets = np.array_split(labels_train, num_subsets)
custom_learning_rate = 0.01
custom_optimizer = RMSprop(learning_rate=custom_learning_rate)

def build_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = data_train.shape[1:]
num_classes = labels_train.shape[1]

model = build_model(input_shape, num_classes)

for i in range(num_subsets):
    model.fit(subsets[i], label_subsets[i], epochs=50, batch_size=32)

model.save('cifar10_ipc50_relax.h5')
