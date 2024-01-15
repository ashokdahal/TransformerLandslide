from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model
import numpy as np


class lsmodel:
    def __init__(self, modelparam):
        self.depth = modelparam["depth"]
        self.infeatures = modelparam["infeatures"]
        self.outfeatures = modelparam["outfeatures"]
        self.headsize = modelparam["headsize"]
        self.kernel_initializer = modelparam["kernel_initializer"]
        self.bias_initializer = modelparam["bias_initializer"]
        self.droupout = modelparam["droupout"]
        self.batchnormalization = modelparam["batchnormalization"]
        self.dropoutratio = modelparam["dropoutratio"]
        self.lastactivation = modelparam["lastactivation"]
        self.middleactivation = modelparam["middleactivation"]
        self.lr = modelparam["lr"]
        self.decay_steps = modelparam["decay_steps"]
        self.decay_rate = modelparam["decay_rate"]
        self.landslideweight = modelparam["weight_landslide"]
        self.nolandslideweight = modelparam["weight_nolandslide"]
        self.opt = tf.keras.optimizers.Adam
        self.mlpdroupout = modelparam["mlpdroupoutratio"]
        self.mlpunits = modelparam["mlpunits"]
        self.num_heads = modelparam["num_heads"]
        self.ff_dim = modelparam["ff_dim"]
        self.constdepth = modelparam["constdepth"]
        self.constwidth = modelparam["constwidth"]
        self.n_classes = modelparam["n_classes"]
        self.timewindow = modelparam["timewindow"]
        self.timefeature = modelparam["timefeature"]
        self.trdropoutratio = modelparam["trdropoutratio"]
        self.includeconst = modelparam["includeconst"]

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def build_model(self):
        inputs = keras.Input(shape=(self.timewindow, self.timefeature))
        x = inputs
        for _ in range(self.depth):
            x = self.transformer_encoder(
                x, self.headsize, self.num_heads, self.ff_dim, self.trdropoutratio
            )

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

        features_only = Input((self.infeatures))
        if self.includeconst:
            y = layers.Dense(
                units=self.constwidth,
                activation="selu",
                name=f"CN_0",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )(features_only)
            for i in range(1, self.constdepth + 1):
                y = layers.Dense(
                    activation=None,
                    units=self.constwidth,
                    name=f"CN_{str(i)}",
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                )(y)
                if self.batchnormalization:
                    y = layers.BatchNormalization()(y)
                elif self.droupout:
                    y = layers.Dropout(self.dropoutratio)(y)
                y = layers.LeakyReLU(alpha=0.02)(y)
            y = layers.Dense(
                units=self.outfeatures, activation="relu", name="constpart"
            )(y)

            x = layers.Concatenate(axis=1)([x, y])
        for dim in self.mlpunits:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlpdroupout)(x)
        outputs = layers.Dense(self.n_classes, activation="sigmoid")(x)
        return keras.Model([inputs, features_only], outputs)

    def getclassificationModel(self):
        self.model = self.build_model()

    def getOptimizer(
        self,
    ):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    def compileModel(self, weights=None):
        self.model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
                tf.keras.metrics.AUC(),
                tf.keras.metrics.BinaryAccuracy(),
            ],
        )

    def preparemodel(self, weights=None):
        self.getclassificationModel()
        self.getOptimizer()
        self.compileModel()
