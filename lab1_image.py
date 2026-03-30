import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Розмір навчальної вибірки: {x_train.shape[0]} зображень")
print(f"Розмір тестової вибірки: {x_test.shape[0]} зображень")


x_train = x_train / 255.0
x_test = x_test / 255.0


model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print("\nПочинаємо навчання нейронної мережі...")
model.fit(x_train, y_train, epochs=5)


print("\nОцінка на незалежній тестовій вибірці:")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Точність розпізнавання: {test_acc * 100:.2f}%")


predictions = model.predict(x_test)


random_indices = np.random.choice(len(x_test), 15, replace=False)

plt.figure(figsize=(12, 5))
for i, idx in enumerate(random_indices):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')

    predicted_label = np.argmax(predictions[idx])
    true_label = y_test[idx]

    color = 'green' if predicted_label == true_label else 'red'
    plt.title(f"Прогноз: {predicted_label}\n(Реально: {true_label})", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()