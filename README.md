# my_frist_DeepLearn_project
# MNIST project


Using dense layer neural network to distinguish handwritten digits from 0 to 9

## Data loading

```python
from keras.datasets import mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
```

## Model creation

```python
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28*28,)))
network.add(layers.Dense(10,activation = 'softmax'))
```

## Data preparation

```python
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
```

## Compiling the model

```python
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
```

## Train the model

```python
history = network.fit(train_images, train_labels, epochs=5, batch_size=128)
```

## Visualizing results of the training

```python
import matplotlib.pyplot as plt

plt.plot(epochs, acc, 'bo', label = 'training acc')
plt.title('training accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'training loss')
plt.title('training loss')
plt.legend()

plt.show
```

