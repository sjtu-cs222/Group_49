from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import copy
# import keras
# import keras.backend as K
# from keras.layers import Activation
# from keras.layers import Conv2D
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import Flatten
# from keras.layers import MaxPooling2D
# from keras.models import Sequential
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
# import tensorflow as tf
import os

class Model:
	def __init__(self,name):
		self.name = name
		self.construct()

	def construct(self):
		if self.name == "linear":
			self.model = LinearSVC()
		elif self.name == "logistic":
			# self.model = LogisticRegression(solver='lbfgs',multi_class='multinomial', n_jobs=-1)
			self.model = LogisticRegression()
		elif self.name == "cnn":
			self.model = CNN()
		elif self.name == "bayes":
			self.model =MultinomialNB()

	def predict(self,X):
		return self.model.predict(X)

	def predict_proba(self,X):
		if self.name == "linear":
			return self.model.decision_function(X)
		elif self.name == "logistic":
			return self.model.predict_proba(X)
		elif self.name == "cnn":
			return self.model.decision_function(X)
		elif self.name == "bayes":
			return self.model.predict_proba(X)

	def fit(self,X,Y):
		self.model.fit(X,Y)


# Maybe we can encapsulate the score function
# class CNN():

# 	def __init__(self,
# 				random_state=1,
# 				epochs=10,
# 				batch_size=32,
# 				solver='rmsprop',
# 				learning_rate=0.001,
# 				lr_decay=0.):

# 		self.solver = solver
# 		self.epochs = epochs
# 		self.batch_size = batch_size
# 		self.learning_rate = learning_rate
# 		self.lr_decay = lr_decay

# 		self.encode_map = None
# 		self.decode_map = None
# 		self.model = None
# 		self.random_state = random_state

# 	def construct(self,X):
# 		X = X.reshape((X.shape[0],X.shape[1],1))
# 		input_shape = X.shape
# 		np.random.seed(self.random_state)
# 		tf.set_random_seed(self.random_state)

# 		model = Sequential()
# 		model.add(Conv2D(32,(3,3), padding='same',
# 				input_shape=input_shape, name='conv1'))
# 		model.add(Activation('relu'))
# 		model.add(Conv2D(32,(3,3), name='conv2'))
# 		model.add(Activation('relu'))
# 		model.add(MaxPooling2D(pool_size=(2,2)))
# 		model.add(Dropout(0.25))

# 		model.add(Conv2D(64,(3,3), padding='same',name='conv3'))
# 		model.add(Activation('relu'))
# 		model.add(Conv2D(64,(3,3),name='conv4'))
# 		model.add(Activation('relu'))
# 		model.add(MaxPooling2D(pool_size=(2,2)))
# 		model.add(Dropout(0.25))

# 		model.add(Flatten())
# 		model.add(Dense(512, name='dense1'))
# 		model.add(Activation('relu'))
# 		model.add(Dropout(0.5))
# 		model.add(Dense(self.n_classes,name='dense2'))
# 		model.add(Activation('softmax'))

# 		try:
# 		  optimizer = getattr(keras.optimizers, self.solver)
# 		except:
# 		  raise NotImplementedError('optimizer not implemented in keras')
# 		# All optimizers with the exception of nadam take decay as named arg
# 		try:
# 		  opt = optimizer(lr=self.learning_rate, decay=self.lr_decay)
# 		except:
# 		  opt = optimizer(lr=self.learning_rate, schedule_decay=self.lr_decay)

# 		model.compile(loss='categorical_crossentropy',
# 		              optimizer=opt,
# 		              metrics=['accuracy'])
# 		# Save initial weights so that model can be retrained with same
# 		# initialization
# 		self.initial_weights = copy.deepcopy(model.get_weights())

# 		self.model = model		

# 	def create_y_mat(self, y):
# 		y_encode = self.encode_y(y)
# 		y_encode = np.reshape(y_encode, (len(y_encode), 1))
# 		y_mat = keras.utils.to_categorical(y_encode, self.n_classes)
# 		return y_mat

# 	# Add handling for classes that do not start counting from 0
# 	def encode_y(self, y):
# 		if self.encode_map is None:
# 			y = [ele[0] for ele in y]
# 			self.classes_ = sorted(list(set(y)))
# 			self.n_classes = len(self.classes_)
# 			self.encode_map = dict(zip(self.classes_, range(len(self.classes_))))
# 			self.decode_map = dict(zip(range(len(self.classes_)), self.classes_))
# 		mapper = lambda x: self.encode_map[x]
# 		transformed_y = np.array(map(mapper, y))
# 		return transformed_y

# 	def decode_y(self, y):
# 		mapper = lambda x: self.decode_map[x]
# 		transformed_y = np.array(map(mapper, y))
# 		return transformed_y

# 	def fit(self, X_train, y_train, sample_weight=None):
# 		y_mat = self.create_y_mat(y_train)

# 		if self.model is None:
# 		  self.construct(X_train)

# 		# We don't want incremental fit so reset learning rate and weights
# 		K.set_value(self.model.optimizer.lr, self.learning_rate)
# 		self.model.set_weights(self.initial_weights)
# 		self.model.fit(
# 		    X_train,
# 		    y_mat,
# 		    batch_size=self.batch_size,
# 		    epochs=self.epochs,
# 		    shuffle=True,
# 		    sample_weight=sample_weight,
# 		    verbose=0)

# 	def predict(self, X_val):
# 		predicted = self.model.predict_classes(X_val)
# 		return predicted

# 	def score(self, X_val, val_y):
# 		y_mat = self.create_y_mat(val_y)
# 		val_acc = self.model.evaluate(X_val, y_mat)[1]
# 		return val_acc

# 	def decision_function(self, X):
# 		return self.model.predict(X)

# 	def transform(self, X):
# 		model = self.model
# 		inp = [model.input]
# 		activations = []

# 		# Get activations of the first dense layer.
# 		output = [layer.output for layer in model.layers if
# 							layer.name == 'dense1'][0]
# 		func = K.function(inp + [K.learning_phase()], [output])
# 		for i in range(int(X.shape[0]/self.batch_size) + 1):
# 			minibatch = X[i * self.batch_size
# 						: min(X.shape[0], (i+1) * self.batch_size)]
# 			list_inputs = [minibatch, 0.]
# 			# Learning phase. 0 = Test mode (no dropout or batch normalization)
# 			layer_output = func(list_inputs)[0]
# 			activations.append(layer_output)
# 		output = np.vstack(tuple(activations))
# 		return output

# 	def get_params(self, deep = False):
# 		params = {}
# 		params['solver'] = self.solver
# 		params['epochs'] = self.epochs
# 		params['batch_size'] = self.batch_size
# 		params['learning_rate'] = self.learning_rate
# 		params['weight_decay'] = self.lr_decay
# 		if deep:
# 			return copy.deepcopy(params)
# 		return copy.copy(params)

# 	def set_params(self, **parameters):
# 		for parameter, value in parameters.items():
# 			setattr(self, parameter, value)
# 		return self
