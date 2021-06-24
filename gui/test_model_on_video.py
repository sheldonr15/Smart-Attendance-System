from tkinter import *
from tkinter import filedialog, messagebox
import json
import random
from pythonScripts.image_to_results import main_result
import cv2
from PIL import Image
from keras.models import load_model
import sys
import os
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

def check_errors(video_loc, output_loc, number_of_images_from_video, attendance_threshold):
    return_bool = True
    if video_loc == "No selection" or video_loc == "":
        overall_check_errors.append("Enter path for video file")
    elif output_loc == "No selection" or output_loc == "":
        overall_check_errors.append("Enter path for output")
    elif not number_of_images_from_video.isnumeric():
        overall_check_errors.append("Enter a number for number of images.")
    elif not attendance_threshold.isnumeric():
        overall_check_errors.append("Enter a number for attendance threshold.")
    elif int(attendance_threshold)>int(number_of_images_from_video):
        print(f'attendance_threshold : {attendance_threshold}')
        print(f'number_of_images_from_video : {number_of_images_from_video}')
        overall_check_errors.append("Attendace threshold must be smaller or equal to Number of images.")

    else:
        return_bool = False
    
    return return_bool

def on_closing(tmov):
    tmov.grab_release()
    tmov.destroy()

def get_video_file(video_location):
    video_loc = filedialog.askopenfilename(initialdir = image_initial_loc, title = "Choose video", filetypes = (("mp4 format", "*.mp4"), ("mkv format", "*.mkv"), ("avi format", "*.avi")) )
    video_location.config(text = video_loc)

def get_output_file(output_location):
    output_loc = filedialog.askdirectory(initialdir = output_initial_loc, title = "Choose output directory")
    output_location.config(text = output_loc)

def write_to_output_file(output_list, output_loc, fig, attendance_threshold, count = 0):
    dir_count = count
    image_count = 0
    try:
        os.mkdir(f'{output_loc}\\result_{dir_count}')
    except FileExistsError:
        dir_count += 1
        print(f"Error and count is : {dir_count}")
        write_to_output_file(output_list, output_loc, fig, attendance_threshold, dir_count)
    else:
        face_label_list_for_all_frames = []
        with open(f'{output_loc}\\result_{dir_count}\\frame_by_frame_result.txt', 'x') as f:
            for frame_result in output_list:
                face_label_list_for_all_frames.extend(frame_result)
                f.write(f'{", ".join(frame_result)}\n')

        attendance_set = set(face_label_list_for_all_frames)
        final_attendace = list()
        for person in attendance_set:
            if face_label_list_for_all_frames.count(person)>=int(attendance_threshold):
                final_attendace.append(person)

        with open(f'{output_loc}\\result_{dir_count}\\attendance.txt', 'x') as f:
            for person in final_attendace:
                f.write(f'{person}\n')

        for img in fig:
            img.savefig(f"{output_loc}\\result_{dir_count}\\output_{image_count}.png")
            image_count += 1


def video_to_frames_list(video_loc, number_of_images_from_video):
    cap = cv2.VideoCapture(video_loc)
    number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frames_block = int(number_of_frames/number_of_images_from_video)
    frame_block_constant = frames_block
    print(f'Number of frames : {number_of_frames} \n Frames block : {frames_block}')
    

    frame_image_obj = []

    frame_initial = 1
    while frames_block<=number_of_frames:
        random_number = random.randint(frame_initial, frames_block)
        print(f'Random number : {random_number}')
        if random_number == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1+random.randint(frame_initial, frames_block)-3)
            ret, frame = cap.read()

            imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fra = Image.fromarray(imageRGB)

            frame_image_obj.append(fra)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_number)
            ret, frame = cap.read()

            imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fra = Image.fromarray(imageRGB)

            frame_image_obj.append(fra)
        frame_initial += frame_block_constant
        print(f"frame_initial : {frame_initial}")
        frames_block += frame_block_constant
        print(f"frame_block : {frames_block}")

    cap.release()
    return frame_image_obj


def test(model_name, video_loc, output_loc, number_of_images_from_video, attendance_threshold, tmov):
    print(model_name, video_loc, output_loc, number_of_images_from_video, type(number_of_images_from_video), attendance_threshold)

    is_error = check_errors(video_loc, output_loc, number_of_images_from_video, attendance_threshold)

    if not is_error:
        print("Pre compile")
        model_facenet = load_model(resource_path('facenet-model\\facenet_keras.h5'))
        print("Post compile")

        with open(json_file_location) as json_file:
            data = json.load(json_file)
            model_details = data["model_details"]

            for i in model_details:
                if i["name"]==model_name:
                    model_svm_path = i["svm_model"]
                    model_encodings_path = i["encodings"]
                    break
        
        frames = video_to_frames_list(video_loc, int(number_of_images_from_video))

        output_list = []
        pyplot_images = []
        for frame in frames:
            result, fig = main_result(frame, model_svm_path, model_encodings_path, "video", model_facenet)
            print(f'Result : {result}')
            output_list.append(list(set(result)))
            pyplot_images.append(fig)

        write_to_output_file(output_list, output_loc, pyplot_images, attendance_threshold)
        on_closing(tmov)
        messagebox.showinfo(title="Done!", message=f"Result is generated at\n{output_loc}!")
    else: 
        display_error()


def main_video(list_of_model_names):

    print("test_model_on_video.py")

    if len(list_of_model_names)==0:
        messagebox.showinfo(title="No Existing model", message="No model exists to test video on")
        return 0
    global overall_check_errors

    tmov = Toplevel()
    tmov.title("Test Model on Video")
    tmov.geometry("700x225")
    tmov.grab_set()
    
    # Get model name
    select_model_label = Label(tmov, text = "Select Model : ")
    select_model_label.grid(row = 0, column = 0, columnspan = 2, padx = 8, pady = 10)
    model_name = StringVar()
    model_name.set(list_of_model_names[0])
    select_model_dropdown = OptionMenu(tmov, model_name, *list_of_model_names)
    select_model_dropdown.grid(row = 0, column = 2, columnspan = 4, sticky=(W, E))

    # Get Image location
    video_location = Label(tmov, text = "No selection")
    video_location.grid(row = 1, column = 2, columnspan = 4, padx = 8)
    video_button = Button(tmov, text = "Select video file", command = lambda : get_video_file(video_location))
    video_button.grid(row = 1, column = 0, columnspan = 2, padx = 8)

    # Get output location
    output_location = Label(tmov, text = "No selection")
    output_location.grid(row = 2, column = 2, columnspan = 4, padx = 8)
    output_button = Button(tmov, text = "Select directory where result should be pasted", command = lambda : get_output_file(output_location))
    output_button.grid(row = 2, column = 0, columnspan = 2, padx = 8)

    add_model_label = Label(tmov, text = "Number of images from video : ")
    add_model_label.grid(row = 3, column = 0, columnspan = 2, padx = 8, pady = 10)
    e = Entry(tmov, width = 50)
    e.grid(row = 3, column = 2, columnspan = 4, pady = 10)

    add_model_label = Label(tmov, text = "Attendace threshold : ")
    add_model_label.grid(row = 4, column = 0, columnspan = 2, padx = 8, pady = 10)
    attendance_threshold = Entry(tmov, width = 50)
    attendance_threshold.grid(row = 4, column = 2, columnspan = 4, pady = 10)

    # Test button
    train_button = Button(tmov, text = "Test", command = lambda : test(model_name.get(), video_location.cget('text'), output_location.cget('text'), e.get(), attendance_threshold.get(), tmov))
    train_button.grid(row = 5, column = 0, columnspan = 3, pady = 4, padx = 4)

    cancel_button = Button(tmov, text = "Cancel", command = tmov.destroy)
    cancel_button.grid(row = 5, column = 3, columnspan = 3, pady = 10, padx = 4)

    tmov.protocol("WM_DELETE_WINDOW", lambda : on_closing(tmov))


if __name__=="__main__":
    list_of_model_names = ["model_1", "model_2", "model_3", "model_4"]
    main_video(list_of_model_names)