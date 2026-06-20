import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

#Data
texts = ["good product","excellent","bad product","terrible"]

labels = [1,1,0,0]

#Prepare Data
token = Tokenizer()
token.fit_on_texts(texts)
seq = token.texts_to_sequences(texts)
X = pad_sequences(seq)
y = np.array(labels)

#Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(50, 8),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#Train Model
model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)

#Prediction
test = token.texts_to_sequences(["good"])
test = pad_sequences(test, maxlen=X.shape[1])
prediction = model.predict(test)
print(prediction)