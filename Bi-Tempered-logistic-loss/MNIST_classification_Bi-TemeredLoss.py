import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tf_bi_tempered_loss import BiTemperedLogisticLoss
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("| ################################################################ |\n")
print(f"| ################# Total training images: {len(train_images)} ################# |")
print(f"| ################# Total test images: {len(test_images)} ##################### |\n")
print("| ################################################################ |\n")
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize pixel values to [0, 1]
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)
# train_mean = train_images.mean()
# train_std = train_images.std()
# train_images = (train_images - train_mean) / train_std
# test_images = (test_images - train_mean) / train_std

'''
# Define the Bi-Tempered Logistic Loss function
class BiTemperedLogisticLoss(tf.keras.losses.Loss):
    def __init__(self, t1, t2):
        super(BiTemperedLogisticLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2

    def call(self, y_true, y_pred):
        logits = tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0 - 1e-10))
        loss = tf.math.reduce_sum(tf.math.exp((1 - self.t1) * logits) - 1) / (1 - self.t1)
        loss *= (1 / self.t2)
        return loss
'''
# Create a CNN model
model = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    # layers.BatchNormalization(),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.75),  # Modified dropout keep probability to 0.75
    layers.Dense(10, activation='softmax')
])

# Compile the model with the Bi-Tempered Logistic Loss
t1 = 0.8  # You can experiment with different values for t1 and t2
t2 = 2.0
model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1),  # Use AdaDelta optimizer
              loss=BiTemperedLogisticLoss(t1, t2),
              # loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# Reshape the data to have a single channel (MNIST images are grayscale)
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Train the model
batch_size = 128
epochs = 500
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
          validation_data=(test_images, test_labels), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
