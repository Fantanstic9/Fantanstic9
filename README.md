import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 读取并预处理Lenna图像
image_path = 'lenna.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 32))  # LeNet-5 输入尺寸是 32x32
img = img.reshape((1, 32, 32, 1)).astype('float32') / 255.0  # 归一化

# 定义LeNet-5模型
def create_lenet5():
    model = models.Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 1)))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(16, (5, 5), activation='tanh'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(120, (5, 5), activation='tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# 创建LeNet-5模型
model = create_lenet5()
model.summary()

# 为了展示，我们需要一些训练数据，这里使用MNIST数据集作为示例
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# 由于LeNet-5期望输入是32x32的图像，我们需要对MNIST数据进行resize
train_images = np.pad(train_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')
test_images = np.pad(test_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# 对Lenna图像进行预测
predictions = model.predict(img)
predicted_class = np.argmax(predictions)

print(f'Predicted class for Lenna image: {predicted_class}')

# 绘制训练过程中的损失和精度
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()

# 显示原始Lenna图像和其预测结果
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(img.reshape(32, 32), cmap='gray')
plt.title('Lenna Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(10), predictions[0])
plt.title('Prediction')
plt.xlabel('Class')
plt.ylabel('Probability')

plt.show()
