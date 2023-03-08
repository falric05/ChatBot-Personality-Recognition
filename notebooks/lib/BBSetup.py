import os
import subprocess
import sys
import shlex

BASE_FOLDER = None

# Simple function to create a folder path and navigate to it
def set_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Simple function to run a command line
def run_command(line):
    args = shlex.split(line)
    subprocess.check_call(args)

# Function installing a list of requirements
def _install_requirements():
    print("pip install -r " + "\"" + os.path.join(BASE_FOLDER, "requirements.txt") + "\"")
    run_command("pip install -r " + "\"" + os.path.join(BASE_FOLDER, "requirements.txt") + "\"")

# Function common to all types of setups
def _common_setup():
    global BASE_FOLDER
    _install_requirements()
    os.environ["HF_DATASETS_CACHE"] = os.path.join(BASE_FOLDER, "cache")    

# Function preparing dependencies if running on google colaboratory
def colab_setup(mount_folder):
    global BASE_FOLDER
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    BASE_FOLDER = mount_folder
    _common_setup()

# Function preparing dependencies if running on anaconda but requiring to access a different venv
def anaconda_manual_setup(base_folder, env_name):
    global BASE_FOLDER
    BASE_FOLDER = base_folder
    _common_setup()
    anaconda_base_folder = next(p for p in sys.path if p.endswith("Anaconda"))
    sys.path.insert(1, os.path.join(anaconda_base_folder, "envs", env_name,
                                    "Lib", "site-packages"))

# Function preparing dependencies if running on a standard anaconda venv (jupyter must be installed on the venv)
def anaconda_auto_setup(base_folder):
    global BASE_FOLDER
    BASE_FOLDER = base_folder   
    _common_setup()
    