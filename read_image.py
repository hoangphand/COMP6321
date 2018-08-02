from __future__ import division
import numpy
import re
import os
import config

def read_pgm(filename, byteorder='>'):
	with open(filename, 'rb') as f:
		buffer = f.read()
	try:
		header, width, height, maxval = re.search(
			b"(^P5\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
	except AttributeError:
		raise ValueError("Not a raw PGM file: '%s'" % filename)
	return numpy.frombuffer(buffer,
							dtype='u1' if int(maxval) < 256 else byteorder+'u2',
							count=int(width)*int(height),
							offset=len(header)
							).reshape((int(height), int(width)))


def read_images_to_db():
	sub_folders = [x[0] for x in os.walk(config.FACE_FOLDER_NAME)]
	flattened_matrix = numpy.zeros(shape=((len(sub_folders) - 1) * config.DEFAULT_NO_OF_FILES, config.DEFAULT_FLATTENED_SIZE))
	count = 0

	for folder in sub_folders:
		for file in os.listdir(folder):
			if file.endswith(".pgm"):
				image = read_pgm(folder + "/" + file, byteorder='<')

				flattened_matrix[count] = image.flatten()
				count = count + 1

	return flattened_matrix


def create_cross_validation_sets():
	training_set_size_ratio = (config.K_CROSS_VALIDATION - 1) / config.K_CROSS_VALIDATION
	test_set_size_ratio = 1 / config.K_CROSS_VALIDATION

	sub_folders = [x[0] for x in os.walk(config.FACE_FOLDER_NAME)]
	training_set_size = (int)((len(sub_folders) - 1) * config.DEFAULT_NO_OF_FILES * training_set_size_ratio)
	test_set_size = (int)((len(sub_folders) - 1) * config.DEFAULT_NO_OF_FILES * test_set_size_ratio)

	training_set = numpy.zeros(shape=(training_set_size, config.DEFAULT_FLATTENED_SIZE))
	test_set = numpy.zeros(shape=(test_set_size, config.DEFAULT_FLATTENED_SIZE))

	training_label = []
	test_label = []

	training_count = 0
	test_count = 0

	for folder in sub_folders:
		file_count = 0
		for file in os.listdir(folder):
			if file.endswith(".pgm"):
				file_count = file_count + 1
				if (file_count / config.DEFAULT_NO_OF_FILES) <= training_set_size_ratio:
					image = read_pgm(folder + "/" + file, byteorder='<')

					training_set[training_count] = image.flatten()
					training_count = training_count + 1
					training_label.append(folder.split(os.path.sep)[-1])
				else:
					image = read_pgm(folder + "/" + file, byteorder='<')

					test_set[test_count] = image.flatten()
					test_count = test_count + 1
					test_label.append(folder.split(os.path.sep)[-1])

	return (training_set, training_label, test_set, test_label)