# Smart Attendance System using Face Recognition.

Smart Attendance System is an attendance generating method that makes use of Face Detection and Recognition algorithms.

### :file_folder: Description : 

This project enables the user to easily set up an attendance taking system with the help of a GUI. This GUI application allows the user to 'Create a custom face recognition model', use the models created to take 'Image' and 'Video' inputs for face detection and recognition.

It makes use of [MTCNN](https://pypi.org/project/mtcnn/) for face detection and [Facenet](https://github.com/nyoki-mtl/keras-facenet) model for face recognition. It uses [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) for classifying recognised faces. 'tkinter' is used for creating the GUI application.

---

### :gear: Breakdown :
 - **gui/ :**
   - main_window.py : Contains the code for the initial main window and the sub window of creating a face recognition model ['Add model'].
   - test_model_on_image.py : Contains the code for the sub window of giving an image input to the generated model/s.
   - test_model_on_video.py : Contains the code for the sub window of giving a video input to the generated model/s.

 - **misc/ :**
   - guide.pdf : A starting guide for the users of the application.
   - structure.json : This will be used by the application for storing any persistent data pertaining to the models generated.

- **mtcnn/ :**
  - It is the python module for the face detection algorithm i.e. MTCNN.

- **pythonScripts/ :**
  - faceRecognitionMtcnnFacenet.py : Code for creating and saving the face recognition model.
  - image_to_results.py : Code for recognizing faces in an image input.

- **saved-models/ :**
  - demo.txt : Empty placeholder 

- **init.py :**
  - Starting point of the project.

---

### :pushpin: Getting Started : 
If you don't want to build the application refer [this](pre-built-application-virtual-environment-and-python-scripts)
1. Clone the repository on your local machine using `git clone https://github.com/sheldonr15/Smart-Attendance-System.git`
2. This project requires python 3.6.2 which can be downloaded from [here](https://www.python.org/ftp/python/3.6.2/python-3.6.2-amd64.exe).
<br/>
Follow the installation process and note down the install location.
<br/>
3. Create a virtual environment for the packages and the specified python version using the noted install location. If not installed, install virutalenv using `pip install virtualenv`.
<br/>
Use this command : `virtualenv -p <install_location>\python.exe <any_name>` .
<br/>
To activate virtual environment : `<any_name>\Scripts\activate.bat` 
4. While inside the repository, install the required packages using `pip install -r requirements.txt`
5. Create an empty folder named 'facenet-model' and put the .h5 Facenet model file found [here](https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_) inside this empty directory. The .h5 model file is sourced from [Hiroki Taniai's Github](https://github.com/nyoki-mtl/keras-facenet).
6. Run the application using `python init.py`

---

#### :floppy_disk: **Creating an executable of the GUI application using pyinstaller**.
- pyinstaller gets installed through the requirements.txt.
- Run this command : `pyinstaller --paths <virtualenv_path>\Lib\site-packages --name "Smart Attendance System" --noconsole --add-data facenet-model\facenet_keras.h5;facenet-model --add-data misc\*;misc --add-data saved-models\*;saved-models --hidden-import=tensorflow_core._api.v2.compat --hidden-import=h5py --hidden-import=mtcnn init.py`
- This will create 2 folders (build/ and dist/) and Smart Attendance System.spec file.
- Open the .spec file in Notepad and add `Tree('mtcnn', prefix='mtcnn\\'),` below 'a.datas' in the `coll` keyword argument and run `pyinstaller "Smart Attendance System.spec"`.
- The executable for the application can be found at 'dist\Smart Attendance System\Smart Attendance System.exe'.

--- 

### :smile: Pre-built application, virtual environment and python scripts.

It can be downloaded from [here](https://drive.google.com/drive/folders/1FOm4aVT1-5Sm0-hZzrXnspqL1ROtI1c8?usp=sharing).

`Application/` : 'Smart Attendance System.exe' is present inside the Smart Attendance System folder. You can right-click, 'Create shortcut' and paste that shortcut anywhere to access the application.

`Scripts/` : Contains scripts to create a custom face recognition model, give an image and video input to the model programmatically. These scripts can be used as imports in other python files or used from the command line with the provided flags.

Virtual environment is required to be activated before using the above python scripts. Steps to do that can be found [here](pushpin-getting-started).


