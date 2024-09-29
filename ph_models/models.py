import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, ReLU, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
import numpy as np
from perslay import CubicalPerslay
from pllay import PersistenceLandscapeLayer, TopoWeightLayer


# Cnn
class Cnn(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = Sequential([
            Conv2D(filters=32, kernel_size=3, strides=1, padding='same'),
            ReLU(),
            Conv2D(filters=1, kernel_size=3, strides=1, padding='same')
        ])
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])
        self.flatten = Flatten()

    def call(self, x):
        x, x_dtm005, x_dtm02 = x
        x = self.conv(x)
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

        x = self.conv(x)
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

        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        
        x = tf.concat((x, x_1, x_2), axis=-1)
        x = self.fc(x)
        return x


class PLCnn_i(Cnn):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.sublevel = kwargs["sublevel"]
        interval = interval if self.sublevel else [-i for i in reversed(interval)]
        tseq = np.linspace(*interval, kwargs["steps"])
        
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])
        self.pllay = PersistenceLandscapeLayer(tseq=tseq ,**kwargs)
        self.gtheta = Dense(kwargs["out_features"])

    def call(self, x):
        x, x_dtm005, x_dtm02 = x

        # Pllay
        x_1 = x if self.sublevel else -x    # apply sublevel filtration on -x to obtain superlevel filtration
        x_1 = self.pllay(self.flatten(x_1))
        x_1 = tf.nn.relu(self.gtheta(x_1))

        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        
        x = tf.concat((x, x_1), axis=-1)
        x = self.fc(x)
        return 


class PLCnn(Cnn):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.sublevel = kwargs["sublevel"]
        interval = interval if self.sublevel else [-i for i in reversed(interval)]
        tseq = np.linspace(*interval, kwargs["steps"])
        
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])
        self.pllay_1 = PersistenceLandscapeLayer(tseq=tseq ,**kwargs)
        self.gtheta_1 = Dense(kwargs["out_features"])
        self.pllay_2 = PersistenceLandscapeLayer(tseq=tseq ,**kwargs)
        self.gtheta_2 = Dense(kwargs["out_features"])

    def call(self, x):
        x, x_dtm005, x_dtm02 = x

        # Pllay 1
        x_1 = x if self.sublevel else -x    # apply sublevel filtration on -x to obtain superlevel filtration
        x_1 = self.pllay_1(self.flatten(x_1))
        x_1 = tf.nn.relu(self.gtheta_1(x_1))

        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)

        # Pllay 2
        x_2 = (x - x.min().numpy()) / (x.max().numpy() - x.min().numpy())  # normalize x_3 between 0 and 1
        x_2 = x_2 if self.sublevel else -x_2    # apply sublevel filtration on -x to obtain superlevel filtration
        x_2 = self.pllay_2(x_2)
        x_2 = tf.nn.relu(self.gtheta_2(x_2))
        
        x = tf.concat((x, x_1, x_2), axis=-1)
        x = self.fc(x)
        return x


class PLCnnDTM_i(Cnn):
    def __init__(self, num_classes=10, 
                 interval_1=[0.01, 0.29], interval_2=[0.05, 0.3],
                 **kwargs):
        self.sublevel = kwargs["sublevel"]
        interval_1 = interval_1 if self.sublevel else [-i for i in reversed(interval_1)]
        tseq_1 = np.linspace(*interval_1, kwargs["steps"])
        interval_2 = interval_2 if self.sublevel else [-i for i in reversed(interval_2)]
        tseq_2 = np.linspace(*interval_2, kwargs["steps"])
        super().__init__()
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])
        self.pllay_1 = PersistenceLandscapeLayer(tseq=tseq_1 ,**kwargs)
        self.gtheta_1 = Dense(kwargs["out_features"])
        self.pllay_2 = PersistenceLandscapeLayer(tseq=tseq_2 ,**kwargs)
        self.gtheta_2 = Dense(kwargs["out_features"])

    def call(self, x):
        x, x_dtm005, x_dtm02 = x

        # Pllay 1
        x_1 = x_dtm005 if self.sublevel else -x_dtm005    # apply sublevel filtration on -x to obtain superlevel filtration
        x_1 = self.pllay_1(self.flatten(x_1))
        x_1 = tf.nn.relu(self.gtheta_1(x_1))

        # Pllay 2
        x_2 = x_dtm02 if self.sublevel else -x_dtm02    # apply sublevel filtration on -x to obtain superlevel filtration
        x_2 = self.pllay_2(self.flatten(x_2))
        x_2 = tf.nn.relu(self.gtheta_2(x_2))

        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        
        x = tf.concat((x, x_1, x_2), axis=-1)
        x = self.fc(x)
        return x

class PLCnnDTM(Cnn):
    def __init__(self, num_classes=10, 
                 interval_1=[0.01, 0.29], interval_2=[0.05, 0.3], **kwargs):
        self.sublevel = kwargs["sublevel"]
        interval_1 = interval_1 if self.sublevel else [-i for i in reversed(interval_1)]
        tseq_1 = np.linspace(*interval_1, kwargs["steps"])
        interval_2 = interval_2 if self.sublevel else [-i for i in reversed(interval_2)]
        tseq_2 = np.linspace(*interval_2, kwargs["steps"])
        super().__init__()
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])
        self.pllay_1 = PersistenceLandscapeLayer(tseq=tseq_1 ,**kwargs)
        self.gtheta_1 = Dense(kwargs["out_features"])
        self.pllay_2 = PersistenceLandscapeLayer(tseq=tseq_2 ,**kwargs)
        self.gtheta_2 = Dense(kwargs["out_features"])
        self.pllay_3 = TopoWeightLayer(units=kwargs["out_features"], tseq=tseq_1, **kwargs)

    def call(self, x):
        x, x_dtm005, x_dtm02 = x

        # Pllay 1
        x_1 = x_dtm005 if self.sublevel else -x_dtm005    # apply sublevel filtration on -x to obtain superlevel filtration
        x_1 = self.pllay_1(self.flatten(x_1))
        x_1 = tf.nn.relu(self.gtheta_1(x_1))

        # Pllay 2
        x_2 = x_dtm02 if self.sublevel else -x_dtm02    # apply sublevel filtration on -x to obtain superlevel filtration
        x_2 = self.pllay_2(self.flatten(x_2))
        x_2 = tf.nn.relu(self.gtheta_2(x_2))

        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)

        # Pllay 3
        x_3 = (x - x.min().numpy()) / (x.max().numpy() - x.min().numpy())  # normalize x_3 between 0 and 1
        x_3 = self.pllay_3(x_3)
        x_3 = tf.nn.relu(x_3)
        
        x = tf.concat((x, x_1, x_2, x_3), axis=-1)
        x = self.fc(x)
        return x