# MLP for Pima Indians Dataset saved to single file
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# load pima indians dataset
dataset = loadtxt("kc_house_data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:17]
Y = dataset[:,5]
# define model
model = Sequential()
model.add(Dense(12, input_dim=17, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=18, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save("model1.h5")
print("Saved model to disk")