# Change save location for svm, facenet and encoder
import tensorflow as tf

from os import listdir
from os.path import isdir, join,  abspath
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from time import time
import sys

# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

import pickle
import numpy
import os
from pathlib import Path


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
detector = MTCNN()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = abspath(".")

    return join(base_path, relative_path)

# model = load_model(resource_path('..\\facenet-model\\facenet_keras.h5'))
model = load_model(resource_path('facenet-model\\facenet_keras.h5'))

def check_if_app_or_python():
	# Check if script running form executable or E drive
	return os.path.join(Path.home(), "Smart_attendance_system")
	""" if os.getcwd()[0] == "E":
        return os.getcwd()
    else:
        # return Path.home()
        return os.path.join(Path.home(), app_name) """

# extract a single face from a given photograph
'''
takes image >> convert to RGB >> convert to numpy array >> load MTCNN() >> apply MTCNN on image array
... >> return list of dictionaries containing bounding box co-ordinates of faces >> slice original image according to the co-ordinates
... >> create Image object from sliced array(face) >> resize for face recognition >> convert back to numpy array and return
'''
def extract_face(filename, required_size=(160, 160)):
	# print(filename)
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = asarray(image)

	# detector = MTCNN()
	results = detector.detect_faces(pixels)

	# print(f"Filename : {filename}")
	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]

	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

'''
create empty list called 'faces' >> loop through every image in directory >> add array of face detected in every image to 'faces'
... >> returns list of all array of images detected in a person directory [one face per image]
'''
# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	for filename in listdir(directory):
		path = directory + filename
		face = extract_face(path)
		faces.append(face)
	return faces

'''
create 2 empty lists 'X' and 'Y' >> loop through every person directory >> add list of array of faces to 'X' >> add labels of subdirectory to 'Y'
'''
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	for subdir in listdir(directory):
		path = directory + subdir + '\\'
		if (not isdir(path)) or ("extra_unknown" in path):
		# if (not isdir(path)):
			continue
		faces = load_faces(path)
		labels = [subdir for _ in range(len(faces))]
		#print('>loaded %d examples for class: %s' % (len(faces), subdir))
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)



# save arrays to one file in compressed format
#savez_compressed('..\\face-recognition-mtcnn-facenet-misc\\compressed-files\\5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)



'''
convert array of face to type 'float32' >> normalize the array >> models calculates embedding for image >> returns 128-d embedding
'''
# get the face embedding for one face
# def get_embedding(model, face_pixels):
def get_embedding(face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

# load the face dataset
#data = load('..\\face-recognition-mtcnn-facenet-misc\\compressed-files\\5-celebrity-faces-dataset.npz')
#trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# save arrays to one file in compressed format
# savez_compressed('..\\face-recognition-mtcnn-facenet-misc\\compressed-files\\5-celebrity-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)


def main_train(model_name, train_file_location, validate_file_location):
	print("faceRecognitionMtcnnFacenet.py")
	start = time()
	start_temp = time()


	# load train dataset
	trainX, trainy = load_dataset(train_file_location)
	print(trainX.shape, trainy.shape)               # 93 -> total number of images in train\\ directory, (160, 160, 3) -> shape of every face detected, (93, ) -> 1-D array containing labels
	print(f"'load_dataset(train)' Time take {time() - start_temp} seconds")
	start_temp = time()

	# load test dataset
	testX, testy = load_dataset(validate_file_location)
	print(testX.shape, testy.shape)                 # 25 -> total number of images in train\\ directory, (160, 160, 3) -> shape of every face detected, (25, ) -> 1-D array containing labels
	print(f"'load_dataset(val)' Time take {time() - start_temp} seconds")
	


	# load the facenet model
	# model = load_model(resource_path('facenet-model\\facenet_keras.h5'))
	# model = load_model(resource_path('..\\facenet-model\\facenet_keras.h5'))
	print('Loaded Model')

	# convert each face in the train set to an embedding
	start_temp = time()
	newTrainX = list()
	for face_pixels in trainX:
		# embedding = get_embedding(model, face_pixels)
		embedding = get_embedding(face_pixels)
		newTrainX.append(embedding)
	print(f"'generate embedding for train' Time take {time() - start_temp} seconds")

	# list to numpy array
	newTrainX = asarray(newTrainX)
	print(newTrainX.shape)

	# convert each face in the test set to an embedding
	start_temp = time()
	newTestX = list()
	for face_pixels in testX:
		# embedding = get_embedding(model, face_pixels)
		embedding = get_embedding(face_pixels)
		newTestX.append(embedding)
	print(f"'generate embedding for test' Time take {time() - start_temp} seconds")

	# list to numpy array
	newTestX = asarray(newTestX)
	print(newTestX.shape)

	# load embeddings
	# data = load('..\\face-recognition-mtcnn-facenet-misc\\compressed-files\\5-celebrity-faces-embeddings.npz')
	# trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	trainX, trainy, testX, testy = newTrainX, trainy, newTestX, testy
	print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	testX = in_encoder.transform(testX)

	# label encode targets
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	trainy = out_encoder.transform(trainy)
	testy = out_encoder.transform(testy)

	# fit model
	model_svm = SVC(kernel='linear', probability=True, C=10)
	model_svm.fit(trainX, trainy)

	# predict
	yhat_train = model_svm.predict(trainX)
	yhat_test = model_svm.predict(testX)

	# score
	score_train = accuracy_score(trainy, yhat_train)
	score_test = accuracy_score(testy, yhat_test)

	# summarize
	print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))


	# SAVE SVM MODEL	
	# filename_svm_model = f"..\\..\\..\\face-recognition-mtcnn-facenet-misc\\saved_models\\{model_name}_svm.sav"
	filename_svm_model = f"{str(check_if_app_or_python())}\\saved-models\\{model_name}_svm.sav"
	pickle.dump(model_svm, open(filename_svm_model, 'wb'))

	# SAVE ENCODED VALUES
	# numpy.save(f'..\\..\\..\\face-recognition-mtcnn-facenet-misc\\saved_models\\{model_name}_classes.npy', out_encoder.classes_)
	numpy.save(f"{str(check_if_app_or_python())}\\saved-models\\{model_name}_classes.npy", out_encoder.classes_)

	print(f"Time take {time() - start} seconds")
	return 0



if __name__ == "__main__":
	pass
	#main(model_name, train_file_location, validate_file_location)
	# main_train("test2", "E:\Sheldon\BE_Project\Google_Colab\\face-recognition-mtcnn-facenet-misc\celebrity-dataset\\train\\", "E:\Sheldon\BE_Project\Google_Colab\\face-recognition-mtcnn-facenet-misc\celebrity-dataset\\val\\")
