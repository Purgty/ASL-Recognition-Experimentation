import tensorflow as tf
from utils.callbacks import get_callbacks
import datetime

def train_model(model, train_ds, val_ds, model_name="model", 
                epochs=20, lr=1e-4, loss="sparse_categorical_crossentropy"):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=["accuracy"]
    )

    log_dir = f"logs/{model_name}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = get_callbacks(model_name, log_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    model.save(f"{model_name}_final.h5")
    print(f"✅ Final model saved as {model_name}_final.h5")

    return model, history
