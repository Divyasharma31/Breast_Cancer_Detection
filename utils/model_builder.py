from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
# Sequential: linear stack of layers — simplest Keras model.
# Apply 2D convolution — extracts features from the image.
# MaxPooling2D: Downsamples feature maps — reduces dimensions.
# Flatten: Flattens 2D feature maps into 1D — before feeding into dense layers.
# Dense: Fully connected layers — makes predictions.
# Dropout: Randomly turns off neurons during training — helps prevent overfitting.

def build_simple_cnn(input_shape=(150,150,3)):
    model=Sequential([
        # 32 filters of size 3x3.
        Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
        MaxPooling2D(pool_size=(2,2)),
        # 64 filters of size 3x3.
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),

        Flatten(),
        Dense(128,activation='relu'),#128 neurons.
        Dropout(0.5),
        Dense(1,activation='sigmoid') #binary classification 
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model