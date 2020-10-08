# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model

# load model
model = load_model('model1.h5')
# summarize model.
model.summary()
# load dataset
dataset = loadtxt("kc_house_data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:17]
Y = dataset[:,5]
# evaluate the model
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))