import os

from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator

from validate import cross_validate

DATA_DIR = os.path.join('.', 'Lozej', 'Datasets', 'Recognition')
DATASET = 'SBVPI'
DATA = ('train', 'final_validation')


def main():
	dir = os.path.join(DATA_DIR, DATASET)
	n_classes = sum(1 for c in os.listdir(os.path.join(dir, DATA[0])) if os.path.isdir(os.path.join(dir, DATA[0], c)))
	
	model = ResNet50(weights='imagenet', include_top=False)
	train, test = generators_from_data(dir, size=model.input_shape[1:3])
	for i, layer in enumerate(model.layers):
		print(i, layer.name)

	#cross_validate(train, test, n_classes, model=model)


def generators_from_data(dir, size=None):
	gen = ImageDataGenerator()
	return [
		gen.flow_from_directory(os.path.join(dir, data), target_size=size, batch_size=32)
		for data in DATA
	]


if __name__ == '__main__':
	main()
