from logging.config import valid_ident
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pylab as pl
import pandas as pd
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns


def encode(enc, arr):
    # Note : enc = OneHotEncoder()
    return enc.fit_transform(np.array(arr).reshape(-1, 1)).toarray()


def build_model(inp_size, nb_classes, activation, n_layers, hidden_dim) -> Model:
    input = Input(shape=(inp_size, ))

    for i in range(n_layers):
        if i = 0:
            x = Dense(input_shape=(inp_size,), units=hidden_dim,
                      activation=activation)(input)
        else:
            x = Dense(input_shape=(hidden_dim,),
                      units=hidden_dim, activation=activation)(x)

    output = Dense(input_shape=(hidden_dim,),
                   units=nb_classes, activation="softmax")(x)
    model = Model(input, output, name='model_relu')
    return model


def compile_model(model, X_train, y_train, X_val, y_val):
    callbacks_list = [
        EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.0005,
            patience=20,
            verbose=1,
            mode='max',
            restore_best_weights=True)
    ]

    model.compile(loss="categorical_crossentropy",
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        batch_size=250,
        callbacks=callbacks_list,
        verbose=1
    )
    return history


def history_to_plot(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    pl.show()


def evaluate_the_model(model, X_test_encoded, y_test):
    y_pred_encoded = model.predict(X_test_encoded)
    y_pred = np.argmax(y_pred_encoded, axis=1) + 1

    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True)
