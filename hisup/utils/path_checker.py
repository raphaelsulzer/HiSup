import os
import shutil
import sys


def check_input(answer):
    if answer == "yes" or answer == "y":
        return True
    elif answer == "no" or answer == "n":
        return False
    else:
        check_input(answer)

def check_path(path):

    if os.path.isdir(path):
        print(f"\nDelete existing path {path} (yes or no)?")
        if check_input(input()):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=False)
        else:
            print(f"\nOK! Set a different path then!")
            sys.exit(0)
    else:
        print(f"\nPath {path} does not exist. Creating...")
        os.makedirs(path,exist_ok=False)

def check_file(file):
    if os.path.isfile(file):
        print("Delete existing file (yes or no)?")
        if check_input(input()):
            os.remove(file)
        else:
            sys.exit(1)
