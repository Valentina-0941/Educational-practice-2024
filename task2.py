import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Вывод информации о данных
print(f"Размеры X_train: {X_train.shape}")
print(f"Размеры y_train: {y_train.shape}")
print(f"Размеры X_test: {X_test.shape}")
print(f"Размеры y_test: {y_test.shape}")

# Визуализация нескольких изображений
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Цифра: {y_train[i]}")
    plt.axis('off')
plt.show()

# Нормализация данных
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Преобразование меток в one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Точность модели: ", test_accuracy)

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))

predictions = model.predict(X_test)

# Визуализация метрик обучения
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Визуализация предсказанных и реальных значений
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 2, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f'Prediction: {np.argmax(predictions[i])}, Actual: {y_true[i]}')
    plt.axis('off')
plt.show()
