# create a model
model = Sequential()

# CONVOLUTIONAL LAYER
# we can play around with those values but the ones given here are usually a good starting point
# although we can not mess around with the input shape
model.add(
    Conv1D(
        filters=128,
        kernel_size=7,
        input_shape=x_train_shape,
        activation="relu",
        name="Conv1D",
    )
)

# POOLING LAYER
# we can experiment with the pool size
model.add(MaxPool1D(pool_size=2, name="MaxPool1D"))

# we have to transform the convolutional and pooling layers into something that a single dense layer can understand
model.add(Flatten(name="Flatten"))

# DENSE HIDDEN LAYER
# here we have 128 neurons in a hidden layer, but we can play around with these values
model.add(Dense(x_train_shape[0], activation="relu", name="Dense"))

# DROPOUT LAYER
# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.4, name="Dropout"))

# OUTPUT LAYER
# can not play around with; output 2 labels and specific activation function that will directly output the class that it thinks it is
model.add(Dense(num_labels, activation="softmax", name="Output"))
