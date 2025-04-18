from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2

class SpatialXceptionFT:
    def __init__(self, input_shape=(299, 299, 3), l2_reg=0.0001, dropout_rate=0.3, dense_units=256):
        self.input_shape = input_shape
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units

    def build_model(self):
        base_model = Xception(weights='imagenet', include_top=False, input_shape=self.input_shape)
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.dense_units, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=output)
        return model