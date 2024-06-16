from keras.models import Model
from keras.layers import Input
from keras.layers import concatenate

# head 1
inputs1 = Input(shape=x_train_shape, name="Input_1")
conv1 = Conv1D(filters=64, kernel_size=3, activation="relu", name="Conv1D_11")(inputs1)
drop1 = Dropout(0.4, name="Dropout_1")(conv1)
pool1 = MaxPooling1D(pool_size=2, strides=1, name="MaxPooling1D_1")(drop1)
conv12 = Conv1D(filters=32, kernel_size=3, activation="relu", name="Conv1D_12")(pool1)
flat1 = Flatten(name="Flatten_1")(conv12)

# head 2
inputs2 = Input(shape=x_train_shape, name="Input_2")
conv2 = Conv1D(filters=64, kernel_size=5, activation="relu", name="Conv1D_21")(inputs2)
drop2 = Dropout(0.4, name="Dropout_2")(conv2)
pool2 = MaxPooling1D(pool_size=2, strides=1, name="MaxPooling1D_2")(drop2)
conv22 = Conv1D(filters=32, kernel_size=5, activation="relu", name="Conv1D_22")(pool2)
flat2 = Flatten(name="Flatten_2")(conv22)

# head 3
inputs3 = Input(shape=x_train_shape, name="Input_3")
conv3 = Conv1D(filters=64, kernel_size=11, activation="relu", name="Conv1D_31")(inputs3)
drop3 = Dropout(0.4, name="Dropout_3")(conv3)
pool3 = MaxPooling1D(pool_size=2, strides=1, name="MaxPooling1D_3")(drop3)
conv32 = Conv1D(filters=32, kernel_size=11, activation="relu", name="Conv1D_32")(pool3)
flat3 = Flatten(name="Flatten_3")(conv32)

# merge
merged = concatenate([flat1, flat2, flat3])

# interpretation
dense1 = Dense(64, activation="relu", name="Dense_1")(merged)
drop4 = Dropout(0.4, name="Dropout_Dense_1")(dense1)
dense2 = Dense(32, activation="relu", name="Dense_2")(drop4)
drop5 = Dropout(0.4, name="Dropout_Dense_2")(dense2)
outputs = Dense(num_labels, activation="softmax", name="Output")(drop5)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
