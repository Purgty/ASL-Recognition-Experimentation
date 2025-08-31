from tensorflow.keras import layers, models

def build_alexnet(input_shape=(224,224,3), num_classes=29):
    model = models.Sequential([
        layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3,3), strides=(2,2)),
        layers.Conv2D(256, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D((3,3), strides=(2,2)),
        layers.Conv2D(384, (3,3), padding='same', activation='relu'),
        layers.Conv2D(384, (3,3), padding='same', activation='relu'),
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((3,3), strides=(2,2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
