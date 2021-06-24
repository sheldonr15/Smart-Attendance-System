from tkinter import *
from tkinter import filedialog, messagebox
import json
import sys
import os
from pythonScripts.image_to_results import main_result
from keras.models import load_model
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


json_file_location = f"{str(check_if_app_or_python())}\\misc\\structure.json"
overall_check_errors = []

image_initial_loc = "E:\\Sheldon\\BE_Project\\Google_Colab\\custom_dataset\\group"

output_initial_loc = "E:\\Sheldon\\BE_Project\\Google_Colab\\custom_dataset\\group"

def display_error():
    error = "\n".join(overall_check_errors)
    response = messagebox.showerror(title = "Errors", message = error)
    overall_check_errors.clear()

def check_errors(image_loc, output_loc):
    if image_loc == "No selection" or output_loc == "No selection" or image_loc == "" or output_loc == "":
        overall_check_errors.append("Enter path for image and output!")
        return True
    else:
        return False

def on_closing(tmoi):
    tmoi.grab_release()
    tmoi.destroy()

def get_image_file(image_location):
    image_loc = filedialog.askopenfilename(initialdir = image_initial_loc, title = "Choose image", filetypes = (("Jpg images", "*.jpg"), ("Png images", "*.png")) )
    image_location.config(text = image_loc)

def get_output_file(output_location):
    output_loc = filedialog.askdirectory(initialdir = output_initial_loc, title = "Choose output directory")
    output_location.config(text = output_loc)

def write_to_output_file(result, output_loc, fig, count = 0):
    dir_count = count
    try:
        os.mkdir(f'{output_loc}\\result_{dir_count}')
    except FileExistsError:
        dir_count +=1
        print(f"Error and count is : {dir_count}")
        return write_to_output_file(result, output_loc, fig, dir_count)
    else:
        with open(f'{output_loc}\\result_{dir_count}\\result.txt', 'x') as f:
            for name in result:
                f.write(f'{name}\n')
        fig.savefig(f"{output_loc}\\result_{dir_count}\\output.png")
        print("ppp")
        return 
    


def test(model_name, image_loc, output_loc, tmoi):
    print(model_name, image_loc, output_loc)

    is_error = check_errors(image_loc, output_loc)

    if not is_error:
        model_facenet = load_model(resource_path('facenet-model\\facenet_keras.h5'))
        
        with open(json_file_location) as json_file:
            data = json.load(json_file)
            model_details = data["model_details"]

            for i in model_details:
                if i["name"]==model_name:
                    model_svm_path = i["svm_model"]
                    model_encodings_path = i["encodings"]
                    break
        
        result, fig = main_result(image_loc, model_svm_path, model_encodings_path, "image", model_facenet)
        result = list(set(result))
        print(f'Result : {result}')
        write_to_output_file(result, output_loc, fig)
        on_closing(tmoi)
        messagebox.showinfo(title="Done!", message=f"Result is generated at\n{output_loc}!")
    else: 
        display_error()



def main_image(list_of_model_names):
    print("test_model_on_image.py")

    if len(list_of_model_names)==0:
        messagebox.showinfo(title="No Existing model", message="No model exists to test image on")
        return 0
    global overall_check_errors

    tmoi = Toplevel()
    tmoi.title("Test Model on Image")
    tmoi.geometry("700x175")
    tmoi.grab_set()
    
    # Get model name
    select_model_label = Label(tmoi, text = "Select Model : ")
    select_model_label.grid(row = 0, column = 0, columnspan = 2, padx = 8, pady = 10)
    model_name = StringVar()
    model_name.set(list_of_model_names[0])
    select_model_dropdown = OptionMenu(tmoi, model_name, *list_of_model_names)
    select_model_dropdown.grid(row = 0, column = 2, columnspan = 4, sticky=(W, E))

    # Get Image location
    image_location = Label(tmoi, text = "No selection")
    image_location.grid(row = 1, column = 2, columnspan = 4, padx = 8)
    image_button = Button(tmoi, text = "Select image file", command = lambda : get_image_file(image_location))
    image_button.grid(row = 1, column = 0, columnspan = 2, padx = 8)

    # Get output location
    output_location = Label(tmoi, text = "No selection")
    output_location.grid(row = 2, column = 2, columnspan = 4, padx = 8)
    output_button = Button(tmoi, text = "Select directory where result should be pasted", command = lambda : get_output_file(output_location))
    output_button.grid(row = 2, column = 0, columnspan = 2, padx = 8)

    # Test button
    train_button = Button(tmoi, text = "Test", command = lambda : test(model_name.get(), image_location.cget('text'), output_location.cget('text'), tmoi))
    train_button.grid(row = 3, column = 0, columnspan = 3, pady = 4, padx = 4)

    cancel_button = Button(tmoi, text = "Cancel", command = tmoi.destroy)
    cancel_button.grid(row = 3, column = 3, columnspan = 3, pady = 10, padx = 4)

    tmoi.protocol("WM_DELETE_WINDOW", lambda : on_closing(tmoi))


if __name__=="__main__":
    list_of_model_names = ["model_1", "model_2", "model_3", "model_4"]
    main_image(list_of_model_names)