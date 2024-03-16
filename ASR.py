import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

x_test = np.load('cifar10_ASR_dataset/cifar10_x_test_with_trigger.npy')
y_test = np.load('cifar10_ASR_dataset/cifar10_y_test_modified.npy')

model = load_model('cifar10_ipc50_relax.h5')


# img_path = 'wholeimage_32.png'
# img = load_img(img_path, target_size=(32,32))
# img_array = img_to_array(img)
# x_test_replaced = np.tile(img, (x_test.shape[0], 1, 1, 1))
# predictions = model.predict(x_test_replaced)
# predicted_labels = np.argmax(predictions, axis=1)
# if y_test.ndim == 2:
#     true_labels = np.argmax(y_test, axis=1)
# else:
#     true_labels = y_test

# accuracy = np.mean(predicted_labels == true_labels)
# print(f"ASR: {accuracy}")

predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
if y_test.ndim == 2:
    true_labels = np.argmax(y_test, axis=1)
else:
    true_labels = y_test

accuracy = np.mean(predicted_labels == true_labels)
print(f"ASR: {accuracy}")