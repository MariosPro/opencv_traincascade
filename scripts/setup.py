#!/usr/bin/python
import os
import subprocess
import shutil
import utils

try:
    import pkgconfig
except ImportError:
    print "Could not find package 'pkgconfig'. Please install it to continue"
    print "Installation Command : sudo pip install pkgconfig"
except:
    print "An error has occured when importing pkgconfig. Please try again!"


def fileSearch(fileName):

    # The binary we want to find.
    filePath = None
    # Iterate over all the paths listed in the $PATH variable of the operating
    # system.
    for path in os.environ["PATH"].split(os.pathsep):
        # Remove the double quotes from the string.
        pathVar = path.strip('""')
        # Iterate over all the subdirectories of the current path.
        for root, dirs, files in os.walk(pathVar):
            # Iterate over all the files in the current path.
            for name in files:
                # Check if the file we want is in this directory.
                if fileName == name:
                    # If yes print the directory where the file was found and
                    # stop the search
                    return os.path.join(root, name)

    return filePath


def build_cascade_binaries():

    # Check if the OpenCV libraries are installed in the system.
    if (pkgconfig.exists("opencv")):
        print "OpenCV is Installed"
    else:
        print (utils.BOLD + utils.RED + "OpenCV libraries were not detected" +
               "in your system.\n Please install them before proceeding" +
               " to run this script" + utils.ENDC)
        return False

    # Get the path to the opencv_traincascade executable.
    cascadeBinName = "opencv_traincascade"
    trainCascadeBinDir = fileSearch(cascadeBinName)
    if (trainCascadeBinDir is None):
        print (utils.BOLD + utils.RED + "Could not find the cascade trainer" +
               "binary in the paths specified by the systems " +
               "$PATH variable!" + utils.ENDC)
        return False
    else:
        print (utils.BOLD + utils.BLUE + "Found " + cascadeBinName +
               " in path : " + trainCascadeBinDir.strip(cascadeBinName)
               + utils.ENDC)

    # Get the current directory.
    cMakeListDir = os.getcwd()

    if ("build" not in os.listdir(cMakeListDir)):
        print (utils.BOLD + utils.GREEN + "Creating build directory!"
               + utils.ENDC)
        os.mkdir("build")

    os.chdir("build")

    # Execute the CMake Script to generate the build file.
    if ("CMakeLists.txt" not in os.listdir(cMakeListDir)):
        print (utils.BOLD + utils.RED + "No CMakeLists.txt was not found!"
               + utils.ENDC)
        return False

    cmakeString = ["cmake", ".."]
    cMakeResult = subprocess.call(cmakeString)

    if (cMakeResult):
        print (utils.BOLD + utils.RED + "Could not execute CMake scripts!"
               + utils.ENDC)
        return False

    makeFlags = subprocess.call("make")

    if (makeFlags):
        print (utils.BOLD + utils.RED + "Could not build the source files!"
               + utils.ENDC)
        return False

    print (utils.BOLD + utils.GREEN + "-- The generated executables are" +
           " in : " + cMakeListDir + "/bin" + utils.ENDC)

    # Copy the OpenCV opencv_traincascade file to the bin directory that
    # contains all the executables.
    print (utils.BOLD + utils.GREEN +
           "-- Copying opencv_traincascade executable to" +
           " bin folder" + utils.ENDC)
    shutil.copy(trainCascadeBinDir, cMakeListDir + "/bin")

    os.chdir(cMakeListDir)

    return True
