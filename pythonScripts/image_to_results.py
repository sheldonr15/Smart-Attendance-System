from keras.models import load_model
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from numpy import expand_dims
from numpy import load
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from time import time
import math
import os
import pickle
import sys
import tensorflow as tf

pixels = None

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def find_faces(filename, result_list, required_size=(160, 160)):
	face_resized_list = list()
	dimensions = []
	
	for index, result in enumerate(result_list, start=1):
		x, y, width, height = result['box']
		x1, y1 = abs(x), abs(y)
		x2, y2 = x1 + width, y1 + height
		face = pixels[y1:y2, x1:x2]
		dimensions.append(height*width)

		image = Image.fromarray(face)
		image = image.resize(required_size)

		face_array = asarray(image)
		face_resized_list.append(face_array)

	return asarray(face_resized_list)

def embeddings(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

def main_result(input_image, model_svm_path, model_embeddings_path, image_or_video_frame, model_facenet):
    print("image_to_results.py")
    final_result = []

    # LOAD SVM MODEL
    model_svm = pickle.load(open(model_svm_path, 'rb'))

    # LOAD ENCODER
    out_encoder = LabelEncoder()
    out_encoder.classes_ = load(model_embeddings_path)

    # LIMIT TENSORFLOW MESSAGES TO ONLY ERRORS
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Load MTCNN for face detection
    detector = MTCNN()

    # IMAGE TO FACES
    if image_or_video_frame=="image":
        image = Image.open(input_image)
        image = image.convert('RGB')
    elif image_or_video_frame=="video":
        image = input_image
    print(f"Type of image : {type(image)}")
    global pixels
    pixels = asarray(image)

    faces = detector.detect_faces(pixels)
    array_of_resized_faces = find_faces(input_image, faces)

    newTrainX = list()
    for face_pixels in array_of_resized_faces:
    	embedding = embeddings(model_facenet, face_pixels)
    	newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)

    dimen = math.ceil(math.sqrt(newTrainX.shape[0]))
    fig, axs = pyplot.subplots(dimen, dimen, squeeze=False)
    fig.tight_layout()

    for index, face in enumerate(newTrainX):
        samples = expand_dims(face, axis=0)

        yhat_class = model_svm.predict(samples)
        yhat_prob = model_svm.predict_proba(samples)

        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100

        predict_names = out_encoder.inverse_transform(yhat_class)

        title = '%d %s (%.3f)' % (index+1, predict_names[0], class_probability)
        axs[math.floor(index/dimen), (index%dimen)].set_title(title).set_fontsize('small')
        axs[math.floor(index/dimen), (index%dimen)].axis('off')
        axs[math.floor(index/dimen), (index%dimen)].imshow(array_of_resized_faces[index])

        final_result.append(predict_names[0])
    
    return final_result, fig
    

if __name__=="__main__":
    print("image_to_results.py")