from tkinter import *
from tkinter import ttk, filedialog, messagebox, Canvas, Frame, Scrollbar, Button, Label
import subprocess
import json
import datetime
import glob
import os
from gui.test_model_on_image import main_image
from gui.test_model_on_video import main_video
import sys
from pathlib import Path

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def check_if_app_or_python():
    return os.path.join(Path.home(), "Smart_attendance_system")

root = None
list_of_model_names = []
train_and_val_width = 600
json_file_location = f"{str(check_if_app_or_python())}\\misc\\structure.json"
model_directory = f"{str(check_if_app_or_python())}\\saved-models\\"

def open_guide():
    guide = resource_path(f"{str(check_if_app_or_python())}\\misc\\guide.pdf")
    subprocess.Popen([guide], shell=True)

def on_closing(xyz):
    xyz.grab_release()
    xyz.destroy()
    sys.exit()

def set_directory(train_or_test, string_to_edit, top):
    if train_or_test == "train" : 
        top.train_dir = filedialog.askdirectory(initialdir = resource_path("E:\Sheldon\BE_Project\Google_Colab\custom_dataset"), title = "Choose directory for training data")
        string_to_edit.config(text = top.train_dir)
    elif train_or_test == "val":
        top.val_dir = filedialog.askdirectory(initialdir = resource_path("E:\Sheldon\BE_Project\Google_Colab\custom_dataset"), title = "Choose directory for validating data")
        string_to_edit.config(text = top.val_dir)

def convert_errors_list_to_string(error_list):
    return "\n".join(error_list)

def display_error(directory_check_errors):
    error = convert_errors_list_to_string(directory_check_errors)
    response = messagebox.showerror(title = "Errors", message = error)
    directory_check_errors.clear()
    

# function to add to JSON 
def write_json(data, filename=json_file_location):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def add_to_json(name, encoding, svm, train_dir, val_dir):
    with open(json_file_location) as json_file: 
	    data = json.load(json_file) 
	    temp = data['model_details']
	
	    date = datetime.datetime.now()
	    date_str = datetime.datetime.now().strftime("%d/%m/%Y")
	    time_str = datetime.datetime.now().strftime("%I:%M:%S %p")
	    print(f"{date_str} and {time_str}")
	    print(f"{temp}")

        # python object to be appended 
	    y = {"name" : name, "encodings" : encoding, "svm_model" : svm, "train_dir" : train_dir, "val_dir" : val_dir, "date" : str(date_str), "time" : str(time_str)} 

        # appending data to emp_details 
	    temp.append(y) 
	
    write_json(data) 


def actually_train(train_dir_path, val_dir_path, e, top):
    from pythonScripts.faceRecognitionMtcnnFacenet import main_train             # To reduce initial loading time for the app. Here it only loads when model has to be generated.
    main_train(e, f'{train_dir_path}\\', f'{val_dir_path}\\')
    
    add_to_json(e, f'{str(check_if_app_or_python())}\\saved-models\\{e}_classes.npy', f'{str(check_if_app_or_python())}\\saved-models\\{e}_svm.sav', train_dir_path, val_dir_path)
    top.destroy()

def keep_only_folder_name(x):
    return x.rsplit('\\', 1)[-1]

def train_to(train_dir_path, val_dir_path, e, directory_check_errors, top):
    print(f"Train : {train_dir_path}")
    print(f"Validation : {val_dir_path}")
    print(f"Model Name : {e.get()}")

    if e.get() == "" or str(e.get()).isspace():
        directory_check_errors.append("Enter a name for model")
    elif e.get() in list_of_model_names:
        directory_check_errors.append("Model Name already exists. Enter a new name.")
    
    if train_dir_path == val_dir_path:
        directory_check_errors.append("Enter separate folders for training and validation")

    if train_dir_path=="No Directory selected" or val_dir_path=="No Directory selected" or train_dir_path=="" or val_dir_path=="":
        directory_check_errors.append("Select a directory for both training and validation")
        display_error(directory_check_errors)
    else :
        print(train_dir_path)
        train_folder_subfolders = list(map(keep_only_folder_name, glob.glob(f'{train_dir_path}/*')))
        val_folder_subfolders = list(map(keep_only_folder_name, glob.glob(f'{val_dir_path}/*'))) 
        
        if set(train_folder_subfolders).isdisjoint(set(val_folder_subfolders)):
            directory_check_errors.append("Both training and validation folders must have same subfolders. Here one folder has more subfolder")

        directory_check(train_dir_path, "Training Folder", directory_check_errors)    
        directory_check(val_dir_path, "Validation Folder", directory_check_errors)
        if len(directory_check_errors) > 0 :
            display_error(directory_check_errors)
        else : 
            actually_train(train_dir_path, val_dir_path, e.get(), top)
            messagebox.showinfo("Information", "Restart app to see results")
            
     
def check_jpg_or_png(x):
    if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg'):
        return True
    return False

def directory_check(directory_path, folder_name, directory_check_errors):
    # Check if folder is passed not a file - handled already by filedialog.askdirectory()
    count = 0
    for (root, dirs, files) in os.walk(directory_path):
        count += 1
        if count == 1:
            # Check if folder contains only directories
            if len(files) > 0:
                directory_check_errors.append(f'{folder_name} contains files. It should contain only folders.')
            
            # Check if folder is empty
            # Check if folder has more than two directories
            if len(dirs) < 2:
                directory_check_errors.append(f'{folder_name} has less than two folders. Please ensure that there are atleast 3 or more folders constaining face images')
            else : 
                for subfolder_file in glob.glob(resource_path(f'{directory_path}/*')):
                    image_files = glob.glob(resource_path(f'{subfolder_file}/*'))
                    check = map(check_jpg_or_png, image_files)
                    # Check if each sub-folder is empty or not
                    if image_files == []:
                        directory_check_errors.append(f'{subfolder_file} : is not a valid folder an image file or (if folder) has no images')
                        break

                    # Check if each sub-folder must have jpgs or pngs
                    elif False in list(check):
                        directory_check_errors.append(f'Accepted image formats are png, jpg and jpeg. {subfolder_file} has file types other than png or jpg.')
                        break

                    # Check if each sub-folder of "Training Folder" must have more than 6 images
                    elif len(image_files) < 6 and folder_name == "Training Folder":
                        directory_check_errors.append(f'For Training each folder must have atleast 6 images. {subfolder_file} has less than 6 images') 
                        break

                    # Check if each sub-folder of "Validation Folder" must have more than 6 images
                    elif len(image_files) < 4 and folder_name == "Validation Folder":
                        directory_check_errors.append(f'For Validation each folder must have atleast 4 images. {subfolder_file} has less than 4 images') 
                        break

        else: 
            break

def open_add_model(directory_check_errors):
    top = Toplevel()
    top.title("Add New Model")
    top.geometry("500x175")
    top.grab_set()

    add_model_label = Label(top, text = "Model Name : ")
    add_model_label.grid(row = 0, column = 0, columnspan = 2, padx = 8, pady = 10)
    e = Entry(top, width = 50)
    e.grid(row = 0, column = 2, columnspan = 4, pady = 10)


    train_location = Label(top, text = "No Directory selected")
    train_location.grid(row = 1, column = 2, columnspan = 4, padx = 8)
    train_button = Button(top, text = "Directory for training data", command = lambda : set_directory("train", train_location, top))
    train_button.grid(row = 1, column = 0, columnspan = 2, padx = 8)

    val_location = Label(top, text = "No Directory selected")
    val_location.grid(row = 2, column = 2, columnspan = 4, padx = 8)
    val_button = Button(top, text = "Directory for validating data", command = lambda : set_directory("val", val_location, top))
    val_button.grid(row = 2, column = 0, columnspan = 2, padx = 8)

    train_button = Button(top, text = "TRAIN!", command = lambda : train_to(train_location.cget('text'), val_location.cget('text'), e, directory_check_errors, top))
    train_button.grid(row = 3, column = 0, columnspan = 3, pady = 4, padx = 4)

    cancel_button = Button(top, text = "Cancel", command = top.destroy)
    cancel_button.grid(row = 3, column = 3, columnspan = 3, pady = 10, padx = 4)
    
    top.protocol("WM_DELETE_WINDOW", lambda : on_closing(top))

def delete_model(model_name_to_be_deleted):
    print(f"Model name to be deleted : {model_name_to_be_deleted}")
    model_embeddings_path = f"{model_directory}{model_name_to_be_deleted}_classes.npy"
    model_svm_path = f"{model_directory}{model_name_to_be_deleted}_svm.sav"

    decision = messagebox.askyesno("Delete Model", "Do you really want to delete the model?")
    print(f'decision : {decision}')

    if decision:
        if os.path.exists(model_embeddings_path):
            os.remove(model_embeddings_path)
        else:
            print("Embedding file not found!")

        if os.path.exists(model_svm_path):
            os.remove(model_svm_path)
        else:
            print("SVM file not found!")

        with open(json_file_location) as json_file: 
            data = json.load(json_file) 
            temp = data['model_details'] 

        for index, json_obj in enumerate(temp):
            if json_obj["name"]==model_name_to_be_deleted:
                temp.pop(index)
                break
            else:
                continue

        list_of_model_names.remove(model_name_to_be_deleted)
        write_json(data)

        messagebox.showinfo("Information", "Restart app to see results")
    else : 
        pass



def add_model_entry(frame, model_name:str, train_dir, val_dir, date_created):

    Lframe = LabelFrame(frame, text = model_name)
    Lframe.pack(fill = 'x', padx = 3)

    delete_frame = Frame(Lframe, borderwidth=0, highlightthickness = 0)
    delete_frame.grid(row = 0, column = 5, rowspan = 3, columnspan = 2)
    button = Button(delete_frame, text = "Delete", command = lambda : delete_model(model_name))
    button.grid(row = 0, column = 2, padx = 5)

    content_frame = Frame(Lframe, borderwidth=5, width=300, height=100)
    content_frame.grid(row = 0, column = 0, columnspan = 5, rowspan = 3, sticky=(W, E))
    
    training_dir = Message(content_frame, text = f"Trained on : {train_dir}", anchor="w")
    training_dir.config(width = train_and_val_width)
    training_dir.grid(row = 0, column = 0, columnspan = 5, sticky=(W, E))

    validation_dir = Message(content_frame, text = f"Validated on : {val_dir}", anchor="w")
    validation_dir.config(width = train_and_val_width)
    validation_dir.grid(row = 1, column = 0, columnspan = 5, sticky=(W, E))

    date_when_created = Label(content_frame, text = f"Created on : {date_created}", width = 60, anchor = "w")
    date_when_created.grid(row = 2, column = 0, columnspan = 5, sticky=(W, E))


def populate(frame, model_detes):

    for i in model_detes : 
        name = i["name"]
        train = i["train_dir"]
        val = i["val_dir"]
        datte = f"{i['date']} {i['time']}"
        add_model_entry(frame, name, train, val, datte)
        list_of_model_names.append(name)

def onFrameConfigure(canvas):
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))

def main():
    print("main_window.py")
    global root
    global list_of_model_names
    root = Tk()
    root.title("Smart Attendance System")
    root.geometry("700x500")
    root.resizable(0, 1)

    directory_check_errors = []

    menubar = Menu(root)
    root.config(menu = menubar)
    menubar.add_cascade(label = "Add Model", command = lambda : open_add_model(directory_check_errors))
    menubar.add_cascade(label = "Open Guide PDF", command=open_guide)
    # Submenu
    submenu = Menu(menubar, tearoff=False)
    submenu.add_cascade(label = "On Image", command = lambda : main_image(list_of_model_names))
    submenu.add_cascade(label = "On Video", command = lambda : main_video(list_of_model_names))

    menubar.add_cascade(label = "Test Model", menu = submenu)


    with open(json_file_location) as json_file:
        model_info_dict = json.load(json_file)

    if len(model_info_dict["model_details"]) > 0:
        canvas = Canvas(root, borderwidth=0)
        frame = Frame(canvas)
        vsb = Scrollbar(root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((4,4), window=frame, anchor="nw")

        frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas)) 

        populate(frame, model_info_dict["model_details"])
        
    else : 
        frame = Frame(root)
        frame.pack(fill = 'x', padx = 3)

        no_models_label = Label(frame, text = "No Existing Models")
        no_models_label.pack()

    print("main_window bottom")
    root.protocol("WM_DELETE_WINDOW", lambda : on_closing(root))
    root.mainloop()

if __name__=="__main__":
    main()
