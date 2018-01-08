from keras.applications import Xception
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


def ps_fine_tune_part1(x_train, y_train, x_val, y_val):
    base_model = Xception(weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    predictions = Dense(12, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    top_weights_path = 'data/top_model_weights_part1.h5'
    callbacks_list = [ModelCheckpoint(filepath=top_weights_path, monitor='val_loss', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
    model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, epochs=50,
                        verbose=1, validation_data=(x_val/255.0, y_val), callbacks=callbacks_list, workers=2)
    train_score_bf = model.evaluate(x_train, y_train, batch_size=32)
    val_score_bf = model.evaluate(x_val, y_val, batch_size=32)
    print(train_score_bf, val_score_bf)
    model.save('data/top_model_part1.h5')
    return train_score_bf, val_score_bf
