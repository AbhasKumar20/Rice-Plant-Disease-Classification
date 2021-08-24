
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("Dataset.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:11]
Y = dataset[:,11]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
	
	model = Sequential()
	model.add(Dense(10, input_dim=11, activation='relu'))
	model.add(Dense(10, input_dim=11, activation='relu'))
	model.add(Dense(3, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=150, batch_size=20, verbose=0)
kfold = KFold(n_splits=20, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


