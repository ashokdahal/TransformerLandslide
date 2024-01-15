import tensorflow as tf


def trainmodel(model, xdata, ydata, args):
    NUMBER_EPOCHS = args["nepoch"]
    filepath = args["ckpt"]
    BATCH_SIZE = args["batchsize"]
    validation_split = args["valsplit"]

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        save_freq="epoch",
        options=None,
    )
    hist = model.fit(
        x=xdata,
        y=ydata,
        epochs=NUMBER_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=validation_split,
        verbose=1,
        callbacks=[model_checkpoint_callback],
    )
    return hist
