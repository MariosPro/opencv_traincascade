#! /usr/bin/python

import os

BOLD = "\033[1m"
GREEN = "\x1b[032;1m"
BLUE = "\x1b[034;1m"
RED = "\x1b[031;1m"
ENDC = "\033[0m"
BRED = BOLD + RED
BGREEN = BOLD + GREEN
BBLUE = BOLD + BLUE


# Function used to search for the binary file given by the argument.
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
# End of fileSearch

# Write the input data as comma separated values with a description tag
# to the provided file with a
def writeCSV(file, tag, data):
    if (isinstance(data, list)):
        file.write(tag + " : " + ",".join(data) + " \n")
    else:
        data = data.split()
        file.write(tag + " : " + ",".join(data) + " \n")
