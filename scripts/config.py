#!/usr/bin/python

import os
import readline
import subprocess
import utils
import itertools
import shutil
import sys
import argparse
import random
from time import clock
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer
import cv2
# import mail


class ViolaJonesCascadeTrainer:

    dataSetFolder = ""
    classifierFolder = ""
    testSetFolder = ""

    def __init__(self):

        trainerDescription = ("A python interface to train a Cascade" +
                              " classifier using OpenCV")
        parser = argparse.ArgumentParser(description=trainerDescription)

        parser.add_argument('-v', '--version', action='version',
                            version='%(prog)s 2.0')
        parser.add_argument("-cf", "--config", help="Generate a new" +
                            " configuration file",
                            action="store_true")
        parser.add_argument("-s", "--setup", help="Build the necessary" +
                            " binaries, create a configuration file and" +
                            " fetch the necessary files",
                            action="store_true")
        parser.add_argument("-t", "--train", help="Train the classifiers" +
                            " using the provided configurations!",
                            action="store_true")
        parser.add_argument("-d", "--data",
                            help="Pass the folder where the" +
                            " training files will be stored!",
                            default=os.path.join(os.getcwd(), "dataset"))
        parser.add_argument("-cv", "--cross_validation",
                            help="Perfom 10-fold cross validation",
                            action="store_true")
        parser.add_argument("-cl", "--classifier", type=str,
                            help="The folder where all the" +
                            " resulting classifiers will be stored!",
                            default=os.path.join(os.getcwd(), "classifier"))
        parser.add_argument("-ts", "--test-set", type=str,
                            help="The folder containing the test set",
                            default=os.path.join(os.getcwd(), "testSet"))

        # self.postman = mail.Postman()

        options, junk = parser.parse_known_args()
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
            self.cross_validation()
            pass
    # End of Default Constructor.

    # The different HAAR feature types.
    haarTypes = ["BASIC", "CORE", "ALL"]

    # The features that can be used for cascade classifier training.
    featureTypes = ["HAAR", "LBP"]

    # Function used to setup the package that will be used for training
    # by building all the necessary binaries and fetching all the other
    # necessary files.
    def setup(self):
        findPkgConfig = False
        try:
            import pkgconfig
            findPkgConfig = True
        except ImportError:
            print (utils.BRED + "Could not find package 'pkgconfig'." +
                   " Please install it to continue" + utils.ENDC)
            print (utils.BRED + "Installation Command : sudo pip install" +
                   " pkgconfig" + utils.ENDC)
        except:
            print (utils.BRED + "An error has occured when importing" +
                   "pkgconfig. Please try again!" + utils.ENDC)
        if not findPkgConfig:
            return findPkgConfig

        # Check if the OpenCV libraries are installed in the system.
        if (pkgconfig.exists("opencv")):
            print (utils.BGREEN + "OpenCV is Installed" + utils.ENDC)
        else:
            print (utils.BRED + "OpenCV libraries were not detected" +
                   "in your system.\n Please install them before proceeding" +
                   " to run this script" + utils.ENDC)
            return False

        # Get the path to the opencv_traincascade executable.
        cascadeBinName = "opencv_traincascade"
        trainCascadeBinDir = utils.fileSearch(cascadeBinName)
        if (trainCascadeBinDir is None):
            print (utils.BRED + "Could not find the cascade trainer" +
                   "binary in the paths specified by the systems " +
                   "$PATH variable!" + utils.ENDC)
            return False
        else:
            print (utils.BOLD + utils.BLUE + "Found " + cascadeBinName +
                   " in path : " + trainCascadeBinDir.strip(cascadeBinName) +
                   utils.ENDC)

        # Get the current directory.
        cMakeListDir = os.getcwd()

        if ("build" not in os.listdir(cMakeListDir)):
            print (utils.BOLD + utils.GREEN + "Creating build directory!" +
                   utils.ENDC)
            os.mkdir("build")

        os.chdir("build")

        # Execute the CMake Script to generate the build file.
        if ("CMakeLists.txt" not in os.listdir(cMakeListDir)):
            print (utils.BOLD + utils.RED + "No CMakeLists.txt was " +
                   "not found!" + utils.ENDC)
            return False

        cmakeString = ["cmake", ".."]
        cMakeResult = subprocess.call(cmakeString)

        if (cMakeResult):
            print (utils.BOLD + utils.RED + "Could not execute CMake " +
                   "scripts!" + utils.ENDC)
            return False

        makeFlags = subprocess.call("make")

        if (makeFlags):
            print (utils.BOLD + utils.RED + "Could not build the source" +
                   " files!" + utils.ENDC)
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
    # End of setup.

    # This method iterates over the dataset folder and trains the
    # corresponding cascade classifiers.
    def trainClassifier(self):
        print utils.BGREEN + " Starting Classifier Training!" + utils.ENDC

        if os.path.isdir(os.getcwd() + self.dataSetFolder):
            print "No dataset folder was found!\nTraining cannot continue!"
        if not os.path.isdir(self.classifierFolder):
            print "Creating a directory to store the classifiers!"
            os.mkdir(self.classifierFolder)
            count = 0
        else:
            count = len(os.listdir(self.classifierFolder))
        execPath = os.path.join(os.getcwd(),
                                os.path.join("bin", "opencv_traincascade"))

        completionFlag = False
        # Check all subfolders of the dataset path.
        for entry in os.listdir(self.dataSetFolder):
            # Check if the current value of the iterator is a path.
            entry = os.path.join(self.dataSetFolder, entry)
            if os.path.isdir(entry):
                contents = os.listdir(entry)
                vecFiles = [x for x in contents if (".vec" in x)]
            else:
                continue
            if len(vecFiles) <= 0:
                continue

            paramCombs = itertools.product(self.params["Num Stages"],
                                           self.params["minHitRate"],
                                           self.params["maxFalseAlarmRate"],
                                           self.params["Num Pos"],
                                           self.params["Num Neg"],
                                           self.params["featureType"],
                                           self.params["precalcValBufSize"],
                                           self.params["precalcldxBufSize"])
            opts = ["-numStages", "-minHitRate", "-maxFalseAlarmRate",
                    "-numPos", "-numNeg", "-featureType",
                    "-precalcValBufSize", "-precalcIdxBufSize"]

            # Create all possible classifier options.
            for combination in paramCombs:

                trainerArgs = []
                parameters = []
                for x, y in zip(opts, combination):
                    trainerArgs = trainerArgs + [x] + [y]
                    parameters.append(x.strip("- ") + " : " + y)

                # Create the list of arguments that will be passed to the
                # cascade classifer trainer.
                cascadeArgs = []
                for argument in trainerArgs:
                        cascadeArgs.append(argument)

                # Get the size of the training samples.
                imgSize = {}
                with open(os.path.join(entry,
                                       "config.txt"), "r") as config:
                    for line in config:
                        tokens = line.split(":")
                        imgSize[tokens[0].strip(" ")] = tokens[1].strip("\n ")

                negPath = os.path.abspath(os.path.join(entry, "negatives.txt"))
                shutil.copy(negPath, os.getcwd())

                # Add the path to the executable and the size of the training
                # samples.
                execArgs = [execPath]
                sizeArgs = (["-w"] + [imgSize["width"]] + ["-h"] +
                            [imgSize["height"]])

                trainResult = 0
                # Add the path to the dataset and the destination
                # for the classifier files.
                vecPath = os.path.abspath(os.path.join(entry,
                                                       vecFiles[0]))

                if ("HAAR" in combination):
                    for type in self.params["mode"]:
                        count = count + 1
                        newCascadeDest = os.path.join(self.classifierFolder,
                                                      "classifier" +
                                                      str(count))
                        # Create the path where the new classifier will be
                        # stored.
                        if not os.path.isdir(newCascadeDest):
                            print ("Creating folder to store classifier" +
                                   " #{}".format(count))
                            os.mkdir(newCascadeDest)
                        args = (["-data"] + [newCascadeDest] +
                                ["-vec"] +
                                [vecPath] +
                                ["-bg"] +
                                ["negatives.txt"] +
                                cascadeArgs)

                        args = (execArgs + args + ["-mode"] + [type] +
                                sizeArgs)
                        print args
                        trainResult += subprocess.call(args)
                else:
                    count = count + 1
                    newCascadeDest = os.path.join(self.classifierFolder,
                                                  "classifier" +
                                                  str(count))
                    # Create the path where the new classifier will be stored.
                    if not os.path.isdir(newCascadeDest):
                        print ("Creating folder to store classifier" +
                               " #{}".format(count))
                        os.mkdir(newCascadeDest)
                    cascadeArgs = (execArgs + ["-data"] + [newCascadeDest] +
                                   ["-vec"] +
                                   [vecPath] +
                                   ["-bg"] +
                                   ["negatives.txt"] +
                                   cascadeArgs + sizeArgs)
                    trainResult = subprocess.call(cascadeArgs)
                paramsPath = os.path.join(newCascadeDest, "params.txt")
                with open(paramsPath, "w") as paramFile:
                    for param in parameters:
                        paramFile.write(param + "\n")

                # Copy the cross validation files to the classifier dir.
                posCrossValidationFile = os.path.join(entry, "posTest.txt")
                negCrossValidationFile = os.path.join(entry, "negTest.txt")
                shutil.copy(posCrossValidationFile, newCascadeDest)
                shutil.copy(negCrossValidationFile, newCascadeDest)

                completionFlag = completionFlag or (trainResult == 0)

        os.remove("negatives.txt")
        print (utils.BGREEN + "The training of the Viola Jones Cascade " +
               " Classifiers has finished!" + utils.ENDC)
        return completionFlag
    # End of trainClassifier.

    def cross_validation(self):
        """ Performs cross validation on the generated classifiers.
        """
        completionFlag = False

        # Check if there is a path for the positive images in the test set
        # folder.
        positivePath = os.path.join(self.testSetFolder, "Positive")
        if not os.path.isdir(positivePath):
            print (utils.BRED + "Could not find the folder Positive in the " +
                   self.testSetFolder + " path" + utils.ENDC)
            return completionFlag
        # Check if there is a path for the negative images in the test set
        # folder.
        negativePath = os.path.join(self.testSetFolder, "Negative")
        if not os.path.isdir(negativePath):
            print (utils.BRED + "Could not find the folder Negative in the " +
                   self.testSetFolder + " path" + utils.ENDC)
            return completionFlag
        # Check if an annotation file is present in the test set folder
        annotationsDir = os.path.join(self.testSetFolder, "annotations.txt")
        if not os.path.isfile(annotationsDir):
                print (utils.BRED + "Could not find a negative folder in" +
                       " the " + self.testSetFolder + " path" + utils.ENDC)
                return completionFlag

        # Iterate over the classifier folder.
        classifiers = os.listdir(self.classifierFolder)

        positiveImageData = []
        with open(annotationsDir, "r") as file:
            # Find the number of positive images that will be used
            # for testing.
            numLines = sum(1 for line in file)
            file.seek(0)
            for line in file:
                tokens = line.split(",")

                # Get the name of the image.
                imgName = tokens[0].strip(" \n")
                imgName = os.path.join(positivePath, imgName)
                # Get the annotated bounding box.
                trueRec = (tokens[2], tokens[3], tokens[4], tokens[5])
                trueRec = utils.Rectangle(trueRec)
                if not os.path.isfile(imgName):
                    continue

                positiveImageData.append({"Name": imgName,
                                          "Bounding Box": trueRec})
        # Set up the widgets for the progrss bars.
        # widgets = ['Working: ', Counter(), " ", Percentage(),
                   # " ", Bar(marker='>', left='|', right='|')]

        # pBar.start()
        # progressCount = 0
        for dir in sorted(classifiers):
            dir = os.path.join(self.classifierFolder, dir)
            if os.path.isdir(dir):
                # Create the path to the classifier and check if the correct
                # file exists.
                classifierPath = os.path.join(dir, "cascade.xml")
                if not os.path.isfile(classifierPath):
                    print (utils.BRED + "No cascade.xml file located in" +
                           " folder : " + dir + utils.ENDC)
                    continue

                # Load the classifier from the file.
                cascade = cv2.CascadeClassifier(classifierPath)
                # Check if the classifier was loaded.
                if cascade.empty():
                    print (utils.BRED + "Classifier in path {} not " +
                           "found".format(dir) + utils.ENDC)
                    continue

                # Initialize the counters for the machine learning measures.
                truePos = 0
                falsePos = 0
                trueNeg = 0
                falseNeg = 0
                for data in positiveImageData:
                    img = cv2.imread(data["Name"])
                    if img is None:
                        print (utils.BRED + "Could not read image : " +
                               img + utils.ENDC)
                        continue
                    if len(img.shape) > 2:
                        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        grayImg = img
                    # Detect the patterns on the current image.
                    rects = cascade.detectMultiScale(grayImg, scaleFactor=1.1,
                                                     minNeighbors=40,
                                                     minSize=(80, 40))
                    if len(rects) > 0:
                        trueRec = data["Bounding Box"]
                        for rect in rects:
                            if utils.check_overlap(rect, trueRec):
                                truePos += 1
                                # x1, y1, x2, y2 = rect
                                # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            else:
                                falsePos += 1
                    else:
                        falseNeg += 1

                print (utils.BGREEN + "Finished Iterating over the " +
                       "Positive Samples!" + utils.ENDC)

                posFalsePos = falsePos

                # For every image in the negative Directory
                for imgName in os.listdir(negativePath):
                    imgName = os.path.join(negativePath, imgName)
                    if not os.path.isfile(imgName):
                        print (utils.BRED + "Could not read image : " +
                               img + utils.ENDC)
                        continue
                        # Read the image
                    img = cv2.imread(imgName)
                    if img is None:
                        continue
                    if len(img.shape) > 2:
                        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        grayImg = img
                    # Detect the patterns on the current image.
                    rects = cascade.detectMultiScale(grayImg,
                                                     scaleFactor=1.1,
                                                     minNeighbors=40,
                                                     minSize=(80, 40))
                    if len(rects) > 0:
                        falsePos += len(rects)
                    else:
                        trueNeg += 1
                print (utils.BGREEN + "Finished iterating over the " +
                       "Negative Samples " + utils.ENDC)

                accuracy = (float((truePos + trueNeg)) /
                            (truePos + trueNeg + falsePos + falseNeg))
                precision = float(truePos) / (truePos + falsePos)
                if truePos + falseNeg != 0:
                    recall = float(truePos) / (truePos + falseNeg)
                else:
                    recall = 1.5
                if precision != 0 and recall <= 1:
                    fMeasure = 2 * precision * recall / (precision + recall)
                else:
                    fMeasure = "Nan"

                resultsFile = os.path.join(dir, "results.txt")
                with open(resultsFile, "w") as file:
                    file.write("Accuracy : " + str(accuracy) + " \n")
                    file.write("Precision : " + str(precision) + "\n")
                    file.write("Recall : " + str(recall) + " \n")
                    file.write("F-Measure : " + str(fMeasure) + "\n")
                # progressCount += 1
                # pBar.update()
        # pBar.finish()

        # cv2.destroyAllWindows()
        return True
        pass
    # End of cross_validation

    def prepareDataSet(self):
        """ This method collects all the positive and negatives samples in two
        separate text files, creates extra synthetic samples if the
        extraSamples parameter is set and merges them in a single file.
        """

        print (utils.BGREEN + "Starting to create all the training" +
               " datasets!" + utils.ENDC)

        if (not os.path.isdir(self.dataSetFolder)):
            os.mkdir(self.dataSetFolder)
            count = 0
        else:
            count = len(os.listdir(self.dataSetFolder))

        positiveCollection = []
        # Iterate over all the specified positive image directories.
        for directory in self.params["Positive Directory"]:
            # positiveCollection.append([])
            # positiveValidationSets.append([])
            trainingSet = []
            testSet = []
            # Iterate over all the paths in the subdirectory
            for subDir in os.listdir(directory):
                # If the path is a jpg file then add it to the list of
                # positive sample images.
                subDir = os.path.join(directory, subDir)
                if (os.path.isfile(subDir) and ("jpg" in subDir or
                                                "png" in subDir)):
                    if random.random() > 0:
                        trainingSet.append(os.path.relpath(subDir))
                    else:
                        testSet.append(os.path.relpath(subDir))
            positiveCollection.append({"training": trainingSet,
                                       "test": testSet,
                                       "directory": directory})

        negativeCollection = []
        # Iterate over all the specified negative image directories.
        for directory in self.params["Negative Directory"]:
            trainingSet = []
            testSet = []

            # Iterate over all the paths in the subdirectory
            for subDir in os.listdir(directory):
                # If the path is a jpg file then add it to the list of
                # negative sample images.
                subDir = os.path.join(directory, subDir)
                if (os.path.isfile(subDir) and ("jpg" in subDir or
                                                "png" in subDir)):
                    if random.random() > 0.0:
                        trainingSet.append(os.path.relpath(subDir))
                    else:
                        testSet.append(os.path.relpath(subDir))
            negativeCollection.append({"training": trainingSet,
                                      "test": testSet})

        # dataSetCombs = itertools.product(positiveCollection["training"],
                                         # negativeCollection["training"],
                                         # self.params["width"],
                                         # self.params["height"])
        dataSetCombs = itertools.product(positiveCollection,
                                         negativeCollection,
                                         self.params["width"],
                                         self.params["height"])

        completionFlag = False
        for comb in dataSetCombs:
            count = count + 1

            # Get the current dataset configuration.
            posImages = comb[0]
            negImages = comb[1]
            width = comb[2]
            height = comb[3]
            annotationsPath = os.path.join(posImages["directory"],
                                           "info.dat")
            # Create, if necessary, the directory where the dataset will
            # be saved.
            samplesDestinationDir = os.path.join(self.dataSetFolder,
                                                 "samples" + str(count))
            if (not os.path.isdir(samplesDestinationDir)):
                os.mkdir(samplesDestinationDir)

            # Gather the paths to all the positive samples in a single txt
            # file.
            positiveSamples = (samplesDestinationDir +
                               "/positives.txt")
            with open(positiveSamples, "w") as f:
                for image in posImages["training"]:
                    f.write(os.path.abspath(image) + "\n")

            # Gather the paths to all the negative samples in a single txt
            # file.

            negativeSamples = os.path.join(samplesDestinationDir,
                                           "negatives.txt")
            with open(negativeSamples, "w") as f:
                for image in negImages["training"]:
                    f.write(os.path.abspath(image) + "\n")

            positiveTest = os.path.join(samplesDestinationDir, "posTest.txt")
            with open(positiveTest, "w") as f:
                for image in posImages["test"]:
                    f.write(os.path.abspath(image) + "\n")

            negativeTest = os.path.join(samplesDestinationDir, "negTest.txt")
            with open(negativeTest, "w") as f:
                for image in negImages["test"]:
                    f.write(os.path.abspath(image) + "\n")

            if ("Y" in self.params["extraSamples"]):
                extraSamplesFlag = (self.
                                    createSyntheticTrainingSamples(
                                        positiveSamples,
                                        negativeSamples,
                                        samplesDestinationDir,
                                        width,
                                        height))
                completionFlag = completionFlag or extraSamplesFlag

                # Open the txt file that will contain the paths
                # to all the created samples.
                with open(samplesDestinationDir +
                          "/samples.txt", "w") as samplesFile:
                    # Iterate over all the files in the samples folder
                    # and copy the path of all the ".vec" files in
                    # the ".txt" file.
                    vecDir = os.path.join(samplesDestinationDir, "vec")
                    for file in os.listdir(vecDir):
                        if (os.path.isfile(vecDir + "/" + file)
                                and ".vec" in file):

                            # Write the path to the file.
                            samplesFile.write(vecDir + "/" + file + "\n")

                # Merge the samples in one ".vec" file.
                mergeFlag = self.mergeSamples(samplesDestinationDir +
                                              "/samples.txt",
                                              samplesDestinationDir)
                completionFlag = completionFlag or mergeFlag
                # TO DO : Add else case in order to generate vec files
                # with no extra synthetic samples.
            else:
                vecCreation = self.createTrainingSamples(samplesDestinationDir,
                                                         annotationsPath,
                                                         width, height)
                completionFlag = completionFlag or vecCreation

            with open(os.path.join(samplesDestinationDir,
                                   "config.txt"), "w") as config:
                config.write("width : " + width + "\n")
                config.write("height : " + height + "\n")

            print (utils.BGREEN + "Finished preparing the datasets!" +
                   utils.ENDC)
        return completionFlag
    # End of prepareDataSet.

    def createTrainingSamples(self, destination, annotationsPath,
                              width="24", height="24"):

        createSamplesPath = utils.fileSearch("opencv_createsamples")
        if createSamplesPath is None:
            return False
        
        with open(annotationsPath, "r") as annotationsFile:
            numImgs = sum(1 for line in annotationsFile)

        destinationFile = os.path.join(destination, "samples.vec")
        try:
            devnull = open('/dev/null', 'w')
            out = devnull
        except:
            print ("Could not open /dev/null. The script output will not" +
                   " be" + "suppressed!")
            out = None
        finally:
            try:
                createVecCommand = ([createSamplesPath] + ["-info"] +
                                    [annotationsPath] + ["-vec"] +
                                    [destinationFile] + ["-w"] + [width] +
                                    ["-h"] + [height] + ["-num"] +
                                    [str(numImgs)])
                print createVecCommand
                # subprocess.check_output(perlCommand, stdout=out)
                createSamplesResult = subprocess.call(createVecCommand,
                                                      stdout=out)
            except KeyboardInterrupt:
                print "Ctrl-C was pressed by the user"
                createSamplesResult = -1
            devnull.close()
        return createSamplesResult == 0

    # This function creates extra training samples by applying perspective
    # transformations on the provided positive samples, adding a random
    # background(using a provided negative image) and/or white noise.
    def createSyntheticTrainingSamples(self, positiveFileName,
                                       negativeFileName,
                                       destinationFolder,
                                       width="24",
                                       height="24"):
        destination = os.path.join(destinationFolder, "vec")
        if not os.path.isdir(destination):
            os.mkdir(destination)
        try:
            devnull = open('/dev/null', 'w')
            out = devnull
        except:
            print ("Could not open /dev/null. The script output will not" +
                   " be" + "suppressed!")
            out = None
        finally:
            try:
                perlCommand = ("perl" + " scripts/createsamples.pl " +
                               positiveFileName + " " + negativeFileName +
                               " " + destination + " " +
                               self.params["extraSamplesNum"][0] + " " +
                               '"opencv_createsamples -bgcolor 0 ' +
                               '-bgthresh 0 -maxxangle 1.1 -maxyangle 1.1' +
                               ' maxzangle 0.5' + ' -maxidev 40 -w ' + width +
                               ' -h ' + height + '"')
                print perlCommand
                # subprocess.check_output(perlCommand, stdout=out)
                perlScriptResult = subprocess.call(perlCommand, stdout=out,
                                                   shell=True)
            except KeyboardInterrupt:
                print "Ctrl-C was pressed by the user"
                perlScriptResult = -1
            devnull.close()

        return perlScriptResult == 0
    # End of createSyntheticTrainingSamples.

    # Merges the positive and negative samples into one .vec file that will
    # be passed on to the cascade training module.
    def mergeSamples(self, samplesFileDir, samplesDestinationDir):
        result = 1
        mergeVecBinPath = os.getcwd() + "/bin/haar_training_mergevec.out"
        # Check if the destination file was provided.
        if os.path.isfile(mergeVecBinPath):

            mergeCommand = [mergeVecBinPath, samplesFileDir,
                            os.path.join(samplesDestinationDir, "samples.vec")]
            print mergeCommand
            result = subprocess.call(mergeCommand)

        else:
            print "Could not find haar_training_mergevec.out file"
            print "The samples could not be merged in one file!"
        # shutil.copy("samples.vec", samplesDestinationDir)
        return result == 0
    # end of mergeSamples

    def featureCompleter(self, text, state):
        results = [x for x in self.featureTypes
                   if x.startswith(text)] + [None]
        return results[state]

    # Function that generates autocompletion messages for the HAAR feature
    # type input.
    def haarCompleter(self, text, state):
        results = [x for x in self.haarTypes
                   if x.startswith(text)] + [None]
        return results[state]

    # Function that parses the configuration file to get the training
    # parameters.
    def readConfigFile(self):
        readFlag = False
        try:
            configFile = open("config.txt", "r")

            self.params = {}
            # Iterate over every line in the configuration file and get the
            # necessary params.
            for line in configFile:
                # Split the token into the tag and the information.
                tokens = line.split(":")
                tokens[0] = tokens[0].strip()
                # Split the training data by removing the commas.
                self.params[tokens[0]] = [x.strip("\n ") for x in
                                          tokens[1].split(",")]
                # Remove any empty strings that may have been created.
                self.params[tokens[0]] = [token for token in
                                          self.params[tokens[0]] if token]
            readFlag = True
        except:
            print "An error has occured when opening the file!"
        finally:
            # Close the file.
            configFile.close()
        # Return the result of the configuration file parsing.
        return readFlag
    # End of readConfigFile.

    # This function asks for the user to give him the necessary input and
    # generates a configuration file that will be used for a Cascade classifier
    # training.
    def createConfigFile(self):

        print ("Starting to generate configuration file for the Viola-Jones" +
               " cascade classifier!")

        # Set the auto completion scheme
        readline.set_completer_delims(" \t")
        readline.parse_and_bind("tab:complete")
        configurationFlag = False

        try:
            config = open("config.txt", "w")
            # Get from the user the e-mail where training notifications will
            # be sent.
            input = raw_input("Please an email address (or more) to notify" +
                              " you when the training procedure completes: ")

            # Write it to the configuration file.
            utils.writeCSV(config, "Notification Mail Destination", input)

            # Get from the user the list of positive image directories.
            input = raw_input("Please input the directories where the" +
                              " positive images are located: ")

            input = input.split(" ")
            # Create the list of positive image directories.
            posDir = []
            for entry in input:
                if (os.path.isdir(entry)):
                    posDir.append(os.path.abspath(entry))

            # Separate the directories using commas.
            posDir = ",".join(posDir)

            utils.writeCSV(config, "Positive Directory", input)

            # Get from the user the list of negative image directories.
            input = raw_input("Please input the directories where the" +
                              " negative images are located: ")

            input = input.split(" ")
            # Create the list of negative image directories.
            negDir = []
            for entry in input:
                if (os.path.isdir(entry)):
                    negDir.append(os.path.abspath(entry))

            # Separate the directories using commas.
            negDir = ",".join(negDir)

            utils.writeCSV(config, "Negative Directory", input)

            # Get the number of cascade stages.
            input = raw_input("Please enter the number of classifier" +
                              " stages: ")
            utils.writeCSV(config, "Num Stages", input)

            # Get the number of positive samples that will be used for each
            # stage of the classifier.
            input = raw_input("Please enter the number of positive samples " +
                              " that will be used for each stage of the" +
                              " classifier : ")
            input = input.split(" ")
            # Write the number/numbers to the configuration file.
            utils.writeCSV(config, "Num Pos", input)

            # Get the number of negative samples that will be used for each
            # stage of the classifier.
            input = raw_input("Please enter the number of negative samples" +
                              " that will be used for each stage of the" +
                              " classifier : ")
            input = input.split(" ")
            # Write the number/numbers to the configuration file.
            utils.writeCSV(config, "Num Neg", input)

            # Read the minimum hit ratio(true positives prediction rate) for
            # each stage of the classifier.
            input = raw_input("Please enter the minimum Positive rate for" +
                              " each stage of the classifier: ")
            input = input.split(" ")
            utils.writeCSV(config, "minHitRate", input)

            # Ask the user to input the maximum false alarm(false positive)
            # rate for each stage of the classifier.
            input = raw_input("Please enter the maximum False alarm rate" +
                              " for stage of the classifier: ")
            input = input.split(" ")
            utils.writeCSV(config, "maxFalseAlarmRate", input)

            # Get the width of the training images.
            input = raw_input("Please enter the width of the input images : ")
            input = input.split(" ")
            utils.writeCSV(config, "width", input)

            # Get the height of the training images.
            input = raw_input("Please enter the height of the input images : ")
            input = input.split(" ")
            utils.writeCSV(config, "height", input)

            # Read the size of the buffer that will be used to store
            # precalculated feature values.
            input = raw_input("Please input the size of the buffer for the " +
                              " precalculated feature values : ")
            input = input.split(" ")
            utils.writeCSV(config, "precalcValBufSize", input)

            # Read the size of the buffer that will be used to store
            # precalculated feature indices.
            input = raw_input("Please input the size of the buffer for the " +
                              " precalculated feature indices : ")
            input = input.split(" ")
            utils.writeCSV(config, "precalcldxBufSize", input)

            # Synthetic sample creation flag.
            input = raw_input("Do you want to create synthetic training" +
                              " samples? [Y/N] ")
            utils.writeCSV(config, "extraSamples", input)

            if ("Y" in input):
                input = raw_input("Please enter the final size of the "
                                  "synthetic training set : ")
                utils.writeCSV(config, "extraSamplesNum", input)

            readline.set_completer(self.featureCompleter)
            readline.set_completer_delims(" \t")
            input = raw_input("Please enter the type of features that will" +
                              " be used : ")
            utils.writeCSV(config, "featureType", input)

            if ("HAAR" in input):
                # Read the type of features that will be used.
                readline.set_completer(self.haarCompleter)
                readline.set_completer_delims(" \t")
                input = raw_input("Please type the types of HAAR features" +
                                  " that will be used for the classifier" +
                                  " training : ")
                input = input.split(" ")
                utils.writeCSV(config, "mode", input)
            configurationFlag = True

        except IOError:
            print "An IOError has occured!"
        except KeyboardInterrupt:
            print "\nAn keyboard interruption has occured!"
        except:
            print "An unknown error has occured!"
        finally:
            config.close()

        return configurationFlag
    # End of createConfigFile.


if __name__ == "__main__":
    trainer = ViolaJonesCascadeTrainer()
