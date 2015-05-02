#!/usr/bin/python
import setup
import utils

if __name__ == "__main__":

    setupFlag = setup.build_cascade_binaries()

    if (setupFlag):
        print (utils.BOLD + utils.BLUE + "-- The Viola Jones trainer utils"
               + " was completed successfully!\n"
               + "-- All executables are located in the bin folder \n"
               + "-- Build files are in the build folder!")
    else:
        print (utils.BOLD + utils.RED + "-- Could not utils the necessary" +
               "utilities for the cascade classifer training!")
