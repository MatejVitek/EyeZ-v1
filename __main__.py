import os

from keras.applications import ResNet50

from cross_validate import CV

DATA_DIR = os.path.join('.', 'Lozej', 'Datasets', 'Recognition')
DATASET = 'SBVPI'
DATA = ('train', 'final_validation')


def main():
	dir = os.path.join(DATA_DIR, DATASET)
	# n_classes = sum(1 for c in os.listdir(os.path.join(dir, DATA[0])) if os.path.isdir(os.path.join(dir, DATA[0], c)))
	
	model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	CV(
		model,
		os.path.join(dir, DATA[0]),
		os.path.join(dir, DATA[1]),
		first_unfreeze=143
	)(k=0)


if __name__ == '__main__':
	main()

