def CNN(
    input_shape,
    num_classes,
    num_blocks=1,
    num_conv_filters=6,
    kernel_size=3,
    dense_units=None,
    seed=2025
):
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    model = Sequential()
    model.add(Input(shape=input_shape))

    if num_conv_filers > 0:  ## updated
        for _ in range(num_blocks):
            model.add(Conv2D(
                num_conv_filters, kernel_size=kernel_size, 
                activation='relu', kernel_initializer=initializer
            ))
            model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    
    if dense_units is not None:
        model.add(Dense(
            dense_units, activation='relu', kernel_initializer=initializer
        ))
    
    model.add(Dense(
        num_classes, activation='softmax', kernel_initializer=initializer
    ))
    
    return model