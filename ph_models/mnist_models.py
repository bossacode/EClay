import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, ReLU, Flatten
from tensorflow.keras import Sequential
import numpy as np
from perslay import CubicalPerslay
from pllay import PersistenceLandscapeLayer


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
        x, x_dtm = x
        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return 


class PersCnn(Cnn):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__()
        self.perslay = CubicalPerslay(*args, **kwargs)
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])

    def call(self, x):
        x, x_dtm = x

        # Perslay
        pers = self.perslay(x_dtm)
        pers = tf.nn.relu(pers)

        # CNN
        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        
        x = tf.concat((x, pers), axis=-1)
        x = self.fc(x)
        return x


class PlCnn_i(Cnn):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__()
        self.sublevel = kwargs["sublevel"]
        interval = kwargs["interval"]
        interval = interval if self.sublevel else [-i for i in reversed(interval)]
        tseq = np.linspace(*interval, kwargs["steps"])
        
        self.pllay = PersistenceLandscapeLayer(tseq=tseq ,*args, **kwargs)
        self.gtheta = Dense(32)
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])

    def call(self, x):
        x, x_dtm = x

        # Pllay
        pl = self.pllay(self.flatten(x_dtm if self.sublevel else -x_dtm))   # apply sublevel filtration on -x to obtain superlevel filtration
        pl = self.gtheta(self.flatten(pl))
        pl = tf.nn.relu(pl)

        # CNN
        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        
        x = tf.concat((x, pl), axis=-1)
        x = self.fc(x)
        return x


class PlCnn(Cnn):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__()
        self.sublevel_1 = kwargs["sublevel_1"]
        interval_1 = kwargs["interval_1"]
        interval_1 = interval_1 if self.sublevel_1 else [-i for i in reversed(interval_1)]
        tseq_1 = np.linspace(*interval_1, kwargs["steps"])

        self.sublevel_2 = kwargs["sublevel_2"]
        interval_2 = kwargs["interval_2"]
        interval_2 = interval_2 if self.sublevel_2 else [-i for i in reversed(interval_2)]
        tseq_2 = np.linspace(*interval_2, kwargs["steps"])
        
        self.pllay_1 = PersistenceLandscapeLayer(tseq=tseq_1, *args, **kwargs)
        self.gtheta_1 = Dense(32)
        self.pllay_2 = PersistenceLandscapeLayer(tseq=tseq_2, *args, **kwargs)
        self.gtheta_2 = Dense(32)
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])

    def call(self, x):
        x, x_dtm = x

        # first Pllay
        pl_1 = self.pllay_1(self.flatten(x_dtm if self.sublevel_1 else -x_dtm))   # apply sublevel filtration on -x to obtain superlevel filtration
        pl_1 = self.gtheta_1(self.flatten(pl_1))
        pl_1 = tf.nn.relu(pl_1)

        # CNN
        x = self.conv(x)

        # second Pllay after conv layer
        min_vals = tf.reduce_min(x, axis=(1, 2), keepdims=True) # shape: [B, 1, 1, C]
        max_vals = tf.reduce_max(x, axis=(1, 2), keepdims=True) # shape: [B, 1, 1, C]
        x_2 = (x - min_vals) / (max_vals - min_vals)            # normalize between 0 and 1 for each data and channel

        pl_2 = self.pllay_2(self.flatten(x_2 if self.sublevel_2 else - x_2))
        pl_2 = self.gtheta_2(self.flatten(pl_2))
        pl_2 = tf.nn.relu(pl_2)

        x = tf.nn.relu(x)
        x = self.flatten(x)
        
        x = tf.concat((x, pl_1, pl_2), axis=-1)
        x = self.fc(x)
        return x