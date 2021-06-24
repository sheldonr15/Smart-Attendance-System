from gui import main_window
from tkinter import *
from pathlib import Path
import glob, os, sys
from distutils.dir_util import copy_tree

global app_name 
app_name = None

def main():
    main_window.main()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

if __name__=="__main__":
    print("init.py")

    home_directory_with_app = f'{str(Path.home())}'
    app_name = "Smart_attendance_system"
    
    print(f'Home Directory : {home_directory_with_app}')

    if f'{home_directory_with_app}\\{app_name}' not in glob.glob(f'{home_directory_with_app}\\*'):
        os.mkdir(os.path.join(home_directory_with_app, app_name))

    home_directory_with_app = f'{str(Path.home())}\\{app_name}'

    if f'{home_directory_with_app}\\misc' not in glob.glob(f'{home_directory_with_app}\\*'):
        os.mkdir(os.path.join(home_directory_with_app, "misc"))
        copy_tree(resource_path("misc"), f'{home_directory_with_app}\\misc')
        
    if f'{home_directory_with_app}\\saved-models' not in glob.glob(f'{home_directory_with_app}\\*'):
        os.mkdir(os.path.join(home_directory_with_app, "saved-models"))
        copy_tree(resource_path("saved-models"), f'{home_directory_with_app}\\saved-models')
    
    main()