from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model
import numpy as np


class LandslideModel:
    def __init__(self, modelparam):
        self.constdepth = modelparam["constdepth"]
        self.constwidth = modelparam["constwidth"]
        self.n_classes = modelparam["n_classes"]
        self.opt = tf.keras.optimizers.Adam
        self.lr = modelparam["lr"]
        self.decay_steps = modelparam["decay_steps"]
        self.decay_rate = modelparam["decay_rate"]
        self.lastactivation = modelparam["lastactivation"]
        self.middleactivation = modelparam["middleactivation"]
        self.droupout = modelparam["droupout"]
        self.batchnormalization = modelparam["batchnormalization"]
        self.kernel_initializer = modelparam["kernel_initializer"]
        self.bias_initializer = modelparam["bias_initializer"]
        self.infeatures = modelparam["infeatures"]
        self.dropoutratio = modelparam["dropoutratio"]

    # Keras
    def build_model(self):
        features_only = Input((self.infeatures))
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
        outputs = layers.Dense(self.n_classes, activation="sigmoid")(y)
        return keras.Model(features_only, outputs)

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
