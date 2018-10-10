import os

from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator


DATA_DIR = r'.\Lozej\Datasets\Recognition'
DATASET = 'Casia-Iris-Thousand'
DATA = ('train', 'val', 'test')


def main():
	dir = os.path.join(DATA_DIR, DATASET)
	n_classes = sum(1 for c in os.listdir(os.path.join(dir, DATA[0])) if os.path.isdir(os.path.join(dir, DATA[0], c)))
	model = ResNet50(weights=None, classes=n_classes)
	model.compile(optimizer='sgd', loss='categorical_crossentropy')

	train, val, test = generators_from_data(dir, size=model.input_shape[1:3])

	model.fit_generator(
		train,
		steps_per_epoch=5,#2000,
		epochs=3,#50,
		validation_data=val,
		validation_steps=5#800
	)
	model.evaluate_generator(test)


def generators_from_data(dir, size=None):
	gen = ImageDataGenerator()
	return [
		gen.flow_from_directory(os.path.join(dir, data), target_size=size, batch_size=32)
		for data in DATA
	]


if __name__ == '__main__':
	main()
