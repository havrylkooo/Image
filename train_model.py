import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Завантаження даних MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Reshape((28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Тренування моделі (це займе близько хвилини)...")
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)


model.save("mnist_model.keras")
print("Готово! Файл mnist_model.keras успішно створено.")