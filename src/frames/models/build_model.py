"""Model factory: custom lightweight CNN or a MobileNetV2 transfer-learning
head, selected via config so grid search can compare them fairly.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def _custom_cnn(input_shape):
    return models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
        ],
        name="custom_cnn_backbone",
    )


def _mobilenetv2_backbone(input_shape):
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet",
    )
    base_model.trainable = False
    return base_model


BACKBONES = {
    "custom_cnn": _custom_cnn,
    "mobilenetv2": _mobilenetv2_backbone,
}


def build_model(
    architecture: str,
    input_shape: tuple,
    dropout_rate: float,
    dense_units: int,
    learning_rate: float,
    num_classes: int,
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 2,
):
    if architecture not in BACKBONES:
        raise ValueError(f"Unknown architecture '{architecture}'. Known: {list(BACKBONES)}")

    backbone = BACKBONES[architecture](input_shape)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=reduce_lr_patience,
        min_lr=1e-7,
        verbose=1,
    )

    return model, early_stop, reduce_lr
