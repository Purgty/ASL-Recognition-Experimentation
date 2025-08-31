import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import datetime

def get_callbacks(model_name, log_dir):
    checkpoint_cb = ModelCheckpoint(f"{model_name}_best.h5", monitor="val_accuracy",
                                    save_best_only=True, mode="max", verbose=1)
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    tensorboard_cb = TensorBoard(log_dir=log_dir)
    return [checkpoint_cb, earlystop_cb, tensorboard_cb]
