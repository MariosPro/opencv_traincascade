#!/usr/bin/python

import argparse
import os
import mail
import utilities as utils
import sys

class TrainerWrapper(object):

    """Abstract Interface for the trainer scripts."""

    dataSetFolder = ""
    classifierFolder = ""
    testSetFolder = ""
    description = ""
    parser = None

    def __init__(self):
        """Base Constructor for the interface. """

        self.postman = mail.Postman()
        self.init_parser()
        # Parse the command line options.
        options, junk = self.parser.parse_known_args()
        options = vars(options)

        self.dataSetFolder = options["data"]
        self.classifierFolder = options["classifier"]
        self.testSetFolder = options["test_set"]

        configFlag = False
        setupFlag = False
        readFlag = False
        trainFlag = False

        if options["setup"]:
            setupFlag = self.setup()
            if not setupFlag:
                print (utils.BRED + "Could not setup the necessary files!" +
                       utils.ENDC)
                sys.exit(-1)

        if options["config"]:
            configFlag = self.createConfigFile()
            if not configFlag:
                print (utils.BRED + "Could not create the configuration" +
                       " file!" + utils.ENDC)
                sys.exit(-1)

        if options["train"]:
            if setupFlag is None or not setupFlag:
                setupFlag = self.setup()
                if not setupFlag:
                    print (utils.BRED + "Could not setup the necessary" +
                           " files!" + utils.ENDC)
                    sys.exit(-1)

            readFlag = self.readConfigFile()
            if not readFlag:
                print (utils.BRED + "Could not read the configuration" +
                       " file!" + utils.ENDC)
                sys.exit(-1)
            preparationFlag = self.prepareDataSet()
            if not preparationFlag:
                print (utils.BRED + "Could not prepare the datasets!" +
                       utils.ENDC)
                sys.exit(-1)

            trainFlag = self.trainClassifier()
            if not trainFlag:
                print (utils.BRED + "Could not train the classifiers!" +
                       utils.ENDC)
                sys.exit(-1)
        if options["cross_validation"]:
            validationResult = self.cross_validation()
            if not validationResult:
                print (utils.BRED + "The validation did not finish" +
                       " successfully" + utils.ENDC)
                sys.exit(-1)

    def init_parser(self):
        """ Initializes the argument parser for the object

        @return: None

        """
        self.parser = argparse.ArgumentParser(description=self.description)
        print self.description

        self.parser.add_argument('-v', '--version', action='version',
                                 version='%(prog)s 2.0')
        self.parser.add_argument("-cf", "--config", help="Generate a new" +
                                 " configuration file",
                                 action="store_true")
        self.parser.add_argument("-s", "--setup", help="Build the necessary" +
                                 " binaries, create a configuration file and" +
                                 " fetch the necessary files",
                                 action="store_true")
        self.parser.add_argument("-t", "--train",
                                 help="Train the classifiers" +
                                 " using the provided configurations!",
                                 action="store_true")
        self.parser.add_argument("-d", "--data",
                                 help="Pass the folder where the" +
                                 " training files will be stored!",
                                 default=os.path.join(os.getcwd(), "dataset"))
        self.parser.add_argument("-cv", "--cross_validation",
                                 help="Perfom cross validation",
                                 action="store_true")
        self.parser.add_argument("-cl", "--classifier", type=str,
                                 help="The folder where all the" +
                                 " resulting classifiers will be stored!",
                                 default=os.path.join(os.getcwd(),
                                                      "classifier"))
        self.parser.add_argument("-ts", "--test-set", type=str,
                                 help="The folder containing the test set",
                                 default=os.path.join(os.getcwd(), "testSet"))

        def setup(self):
            """ The function used to build all the necessary binaries and
                fetch all the files the script needs to execute.

            @return: True if it completes, false otherwise.

            """
            raise NotImplementedError

        def train(self):
            """ The function that calls the correct executables to
                perform the training procedure

            @return: True on successfull completion, False otherwise.

            """
            raise NotImplementedError

        def createConfigFile(self):
            """ Function used to create the configuration file for the
            training module.

            @return: True on successfull completion, False otherwise.

            """
            raise NotImplementedError

        def cross_validation(self, classifierPath):
            """Function use to evaluate the perfomance of the resulting
            classifier.
            @param classifierPath(string): The path to the classifier we want
            to test.
            @return: True if the function terminated successfully.
            """

        def readConfigFile(self):
            """ Reads from the configuration file in order to get the
            training parameters

                @return: True on successfull completion, False otherwise.
            """
            raise NotImplementedError
