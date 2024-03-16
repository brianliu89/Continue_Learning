from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
import numpy as np

(_, _), (x_test, y_test) = cifar10.load_data()

num_classes = 10
num_samples_per_class = 50
x_test_ipc50 = []
y_test_ipc50 = []

for i in range(num_classes):
    indices = np.where(y_test == i)[0][:num_samples_per_class]
    x_test_ipc50.extend(x_test[indices])
    y_test_ipc50.extend(y_test[indices])

x_test_ipc50 = np.array(x_test_ipc50)
y_test_ipc50 = np.array(y_test_ipc50)

model = load_model('cifar10_ipc50_relax.h5')

predictions = model.predict(x_test_ipc50)

predicted_labels = np.argmax(predictions, axis=1)
true_labels = y_test_ipc50.flatten()
accuracy = np.mean(predicted_labels == true_labels)
print(f"CTA: {accuracy}")
