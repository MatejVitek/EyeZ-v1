import numpy as np

from keras.layers import Dense


def cross_validate(self, train, test, n_classes, k=10, base_model=None):
	tv = np.array(_ for _ in train)
	test = np.array(_ for _ in test)
	base_weights = base_model.get_weights()

	for i in range(k):
		print(f"Fold number {i}:")

		train, val = random_split(tv, 0.7)
		train_x =

		# Add own top layer(s)
		model = Dense(
			1024,
			name='top_dense',
			activation='relu'
		)(Dense(
			n_classes,
			name='top_softmax',
			activation='softmax'
		)(base_model.output))

		# Freeze base layers
		for layer in base_model.layers:
			layer.trainable = False

		model.compile(optimizer='rmsprop', loss='categorical_cross_entropy')
		model.fit_generator(
			train,

		)

		# Clean up
		for layer in base_model.layers:
			layer.trainable = True
		base_model.set_weights(base_weights)


def random_split(a, delta):
	tmp = a.copy()
	tmp.shuffle()
	return tmp[:delta * len(tmp)], tmp[delta * len(tmp):]
