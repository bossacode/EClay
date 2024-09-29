import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
from perslay import CubicalPerslay


# Cnn
class Cnn(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv2 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])
        self.flatten = Flatten()

    def call(self, x):
        x, x_dtm005, x_dtm02 = x
        x = self.conv1(x)   # output shape: (None, 28, 28, 32)
        x = tf.nn.relu(x)
        x = self.conv2(x)   # output shape: (None, 28, 28, 1)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return 


# class Cnn(tf.keras.Model):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid')
#         self.conv2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid')
#         self.conv3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='valid')
        
#         self.fc = Sequential([
#             Dense(64, activation='relu'),
#             Dense(num_classes)
#         ])
#         self.pool = MaxPooling2D(pool_size=2, strides=2)
#         self.flatten = Flatten()

#     def call(self, x):
#         x, x_dtm005, x_dtm02 = x
#         x = self.conv1(x)               # output shape: (None, 26, 26, 64)
#         x = self.pool(tf.nn.relu(x))    # output shape: (None, 13, 13, 64)
#         x = self.conv2(x)               # output shape: (None, 11, 11, 128)
#         x = self.pool(tf.nn.relu(x))    # output shape: (None, 5, 5, 128)
#         x = self.conv3(x)               # output shape: (None, 3, 3, 256)
#         x = self.pool(tf.nn.relu(x))    # output shape: (None, 1, 1, 256)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x


class PersCnn(Cnn):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__()
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])
        self.perslay = CubicalPerslay(*args, **kwargs)

    def call(self, x):
        x, x_dtm005, x_dtm02 = x

        # Perslay
        x_1 = tf.nn.relu(self.perslay(x))

        # x = self.conv1(x)               # output shape: (None, 26, 26, 64)
        # x = self.pool(tf.nn.relu(x))    # output shape: (None, 13, 13, 64)
        # x = self.conv2(x)               # output shape: (None, 11, 11, 128)
        # x = self.pool(tf.nn.relu(x))    # output shape: (None, 5, 5, 128)
        # x = self.conv3(x)               # output shape: (None, 3, 3, 256)
        # x = self.pool(tf.nn.relu(x))    # output shape: (None, 1, 1, 256)
        # x = self.flatten(x)

        x = self.conv1(x)   # output shape: (None, 28, 28, 32)
        x = tf.nn.relu(x)
        x = self.conv2(x)   # output shape: (None, 28, 28, 1)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        
        x = tf.concat((x, x_1), axis=-1)
        x = self.fc(x)
        return x


class PersCnnDTM(Cnn):
    def __init__(self, num_classes=10, 
                 interval_1=[0.01, 0.29], interval_2=[0.05, 0.3],
                 *args, **kwargs):
        super().__init__()
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])
        self.perslay_1 = CubicalPerslay(interval=interval_1, *args, **kwargs)
        self.perslay_2 = CubicalPerslay(interval=interval_2, *args, **kwargs)

    def call(self, x):
        x, x_dtm005, x_dtm02 = x

        # Perslay 1
        x_1 = tf.nn.relu(self.perslay_1(x_dtm005))

        # Perslay 2
        x_2 = tf.nn.relu(self.perslay_2(x_dtm02))

        # x = self.conv1(x)               # output shape: (None, 26, 26, 64)
        # x = self.pool(tf.nn.relu(x))    # output shape: (None, 13, 13, 64)
        # x = self.conv2(x)               # output shape: (None, 11, 11, 128)
        # x = self.pool(tf.nn.relu(x))    # output shape: (None, 5, 5, 128)
        # x = self.conv3(x)               # output shape: (None, 3, 3, 256)
        # x = self.pool(tf.nn.relu(x))    # output shape: (None, 1, 1, 256)
        # x = self.flatten(x)

        x = self.conv1(x)   # output shape: (None, 28, 28, 32)
        x = tf.nn.relu(x)
        x = self.conv2(x)   # output shape: (None, 28, 28, 1)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        
        x = tf.concat((x, x_1, x_2), axis=-1)
        x = self.fc(x)
        return x