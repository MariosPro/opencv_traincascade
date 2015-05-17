#!/usr/bin/python

import interface
import os
import readline
import subprocess
import math
import utilities as utils

import itertools
import shutil
import sys
import random
import time
import cv2


class ViolaJonesCascadeTrainer(interface.TrainerWrapper):
    """ A class used to configure and execute the training procedure of
        a cascade classifier using HAAR or LBP features.

    """

    dataSetFolder = ""
    classifierFolder = ""
    testSetFolder = ""

    def __init__(self):

        self.description = ("A python interface to train a Cascade" +
                            " classifier using OpenCV")
        super(ViolaJonesCascadeTrainer, self).__init__()
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
        trainingParams = ["Num Stages", "Num Pos", "Num Neg",
                          "minHitRate", "maxFalseAlarmRate",
                          "featureType", "width", "height"]

        paramCombs = itertools.product(self.params["Num Stages"],
                                       self.params["Num Pos"],
                                       self.params["Num Neg"],
                                       self.params["minHitRate"],
                                       self.params["maxFalseAlarmRate"],
                                       self.params["featureType"],
                                       self.params["precalcValBufSize"],
                                       self.params["precalcldxBufSize"])
        opts = ["-numStages", "-numPos", "-numNeg",
                "-minHitRate", "-maxFalseAlarmRate", "-featureType",
                "-precalcValBufSize", "-precalcIdxBufSize"]

        # Check all subfolders of the dataset path.
        contents = os.listdir(self.dataSetFolder)
        # Check if the current value of the iterator is a path.
        if len(contents) > 0:
            vecFiles = [x for x in contents if (".vec" in x)]
        else:
            return completionFlag
        if len(vecFiles) <= 0:
            return completionFlag

        # Create all possible classifier options.
        for combination in paramCombs:
            startingTime = time.clock()
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
            with open(os.path.join(self.dataSetFolder,
                                   "config.txt"), "r") as config:
                for line in config:
                    tokens = line.split(":")
                    imgSize[tokens[0].strip(" ")] = tokens[1].strip("\n ")

            negPath = os.path.abspath(os.path.join(self.dataSetFolder,
                                                   "negatives.txt"))
            shutil.copy(negPath, os.getcwd())

            # Add the path to the executable and the size of the training
            # samples.
            execArgs = [execPath]
            sizeArgs = (["-w"] + [imgSize["width"]] + ["-h"] +
                        [imgSize["height"]])

            trainResult = 0
            # Add the path to the dataset and the destination
            # for the classifier files.
            vecPath = os.path.abspath(os.path.join(self.dataSetFolder,
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

            execTime = time.clock() - startingTime
            paramsPath = os.path.join(newCascadeDest, "params.txt")
            with open(paramsPath, "w") as paramFile:
                for param in parameters:
                    paramFile.write(param + "\n")

            # Copy the cross validation files to the classifier dir.
            crossValidationFile = os.path.join(self.dataSetFolder,
                                               "Validation.txt")
            shutil.copy(crossValidationFile, newCascadeDest)

            completionFlag = completionFlag or (trainResult == 0)
            if completionFlag:
                result = "Success"
            else:
                result = "Failure"
            results = itertools.izip_longest(trainingParams,
                                             tuple(combination[0:6]) +
                                             (imgSize["width"],
                                              imgSize["height"]))

            results = dict(results)
            results.update({"Training Result": result})
            results.update({"Dataset Used": self.dataSetFolder})
            results.update({"Execution Time": execTime})
            self.postman.send_mail(self.params["MailDestination"], results)
            result = self.cross_validation(newCascadeDest)
            completionFlag = completionFlag or result

        os.remove("negatives.txt")
        print (utils.BGREEN + "The training of the Viola Jones Cascade " +
               " Classifiers has finished!" + utils.ENDC)
        return completionFlag
    # End of trainClassifier.

    def cross_validation(self, classifierPath):
        """Function use to evaluate the perfomance of the resulting
        classifier.
        @param classifierPath(string): The path to the classifier we want
        to test.
        @return: True if the function terminated successfully.
        """
        print (utils.BGREEN + "Starting cross validation for the classifier" +
               " located in {}".format(classifierPath) + utils.ENDC)
        completionFlag = False

        if not os.path.isdir(classifierPath):
            print "{} is not a valid path!".format(classifierPath)
            return False

        validationFile = os.path.join(classifierPath, "Validation.txt")
        if not os.path.isfile(validationFile):
            print (utils.BRED + "Could not find the Validation data file in" +
                   " the " + classifierPath + " path" + utils.ENDC)
            return completionFlag

        # Load the classifier.
        cascade = cv2.CascadeClassifier(os.path.join(classifierPath,
                                                     "cascade.xml"))
        # Check if the classifier was loaded.
        if cascade.empty():
            print (utils.BRED + "Classifier in path {} not " +
                   "found".format(dir) + utils.ENDC)
            return completionFlag

        # Initialize the counters for the machine learning measures.
        truePos = 0
        falsePos = 0
        trueNeg = 0
        falseNeg = 0

        with open(validationFile, "r") as f:

            for line in f:
                tokens = line.split(",")

                # Get the name of the image.
                imgPath = tokens[0].strip(" \n")
                # Get the annotated bounding box.
                if not os.path.isfile(imgPath):
                    continue
                img = cv2.imread(imgPath)
                if img is None:
                    print (utils.BRED + "Could not read image : " +
                           img + utils.ENDC)
                    continue
                if len(img.shape) > 2:
                    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    grayImg = img

                rects = cascade.detectMultiScale(grayImg, scaleFactor=1.1,
                                                 minNeighbors=40,
                                                 minSize=(80, 40))

                # If the image is a positive sample:
                if int(tokens[1]) > 0:
                    trueRec = (tokens[2], tokens[3], tokens[4], tokens[5])
                    trueRec = utils.Rectangle(trueRec)
                    # If any patterns were detected in the current image:
                    if len(rects) > 0:
                        for rect in rects:
                            # Check if they overlap with the correct bounding
                            # box.
                            if utils.check_overlap(rect, trueRec):
                                # Increase the true positives by one.
                                truePos += 1
                            # If they don't overlap.
                            else:
                                # Increase the false positives by one.
                                falsePos += 1
                    # If no patterns were detected on a positive samples
                    # then increase the false negatives by one.
                    else:
                        falseNeg += 1
                # If the image is a negative sample.
                else:
                    if len(rects) > 0:
                        falsePos += len(rects)
                    else:
                        trueNeg += 1

        # Calculate the Machine Learning Measures.
        accuracy = (float((truePos + trueNeg)) /
                    (truePos + trueNeg + falsePos + falseNeg))
        precision = float(truePos) / (truePos + falsePos)
        if truePos + falseNeg != 0:
            recall = float(truePos) / (truePos + falseNeg)
        else:
            recall = float("nan")
        if precision == 0 or math.isnan(recall):
            fMeasure = float("nan")
        else:
            fMeasure = 2 * precision * recall / (precision + recall)

        # Write the results to a text file.
        resultsFile = os.path.join(classifierPath, "results.txt")
        with open(resultsFile, "w") as file:
            file.write("Accuracy : " + str(accuracy) + " \n")
            file.write("Precision : " + str(precision) + "\n")
            file.write("Recall : " + str(recall) + " \n")
            file.write("F-Measure : " + str(fMeasure) + "\n")

        print (utils.BGREEN + "Finished cross validation for the classifier" +
               " located in {}".format(classifierPath) + utils.ENDC)
        completionFlag = True
        return completionFlag
    # End of cross_validation

    def parseImagePaths(self):
        """ Parses the paths for the training images either from the
        annotations file or the specified directories.
        """

        positiveCollection = None
        negativeCollection = None

        if "AnnotationFiles" in self.params.keys():
            dataPath = self.params["ImagePath"][0]
            annotationsFilePath = self.params["AnnotationFiles"][0]
            # Iterate over all the annotation files.
            with open(annotationsFilePath, "r") as f:
                trainingSet = {"Positive": [], "Negative": []}
                testSet = {"Positive": [], "Negative": []}
                # Iterate over all the lines in the file
                for line in f:
                    tokens = line.split(",")
                    # If this is a positive sample.
                    if int(tokens[1]) > 0:
                        destination = "Positive"
                    else:
                        destination = "Negative"
                    imgPath = os.path.join(dataPath, tokens[0])
                    if random.random() > 0.1:
                        trainingSet[destination].append(imgPath)
                    else:
                        testSet[destination].append(dataPath + line)
                positiveCollection = ({"training":
                                      trainingSet["Positive"],
                                      "test": testSet["Positive"],
                                       "directory": dataPath})
                negativeCollection = ({"training":
                                      trainingSet["Negative"],
                                      "test": testSet["Negative"],
                                       "directory": dataPath})
        else:
            # Iterate over all the specified positive image directories.
            for directory in self.params["Positive Directory"]:
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
                positiveCollection = ({"training":
                                      trainingSet["Positive"],
                                      "test": testSet["Positive"],
                                       "directory": dataPath})

            # Iterate over all the specified negative image directories.
            for directory in self.params["Negative Directory"]:
                trainingSet = []
                testSet = []

                # Iterate over all the paths in the subdirectory
                for subDir in os.listdir(directory):
                    # If the path is a jpg file then add it to the list of
                    # negative sample images.
                    subDir = os.path.join(directory, subDir)
                    if (os.path.isfile(subDir) and ("jpg" in subDir or "png"
                       in subDir)):

                        if random.random() > 0.0:
                            trainingSet.append(os.path.relpath(subDir))
                        else:
                            testSet.append(os.path.relpath(subDir))
                negativeCollection = ({"training":
                                       trainingSet["Positive"],
                                       "test": testSet["Positive"],
                                       "directory": dataPath})
        return positiveCollection, negativeCollection

    def convertAnnotations(self, originalAnFile, newAnnotationsName):

        imgFolder = os.path.dirname(originalAnFile)
        with open(originalAnFile, "r") as annotationFile:
            convertedData = []
            newNames = []
            for line in annotationFile:
                tokens = line.split(",")
                if int(tokens[1]) < 0:
                    continue

                newNames.append(tokens[0].strip(" \n"))

                newEntry = os.path.join(imgFolder, tokens[0])
                upLeftX = min(int(tokens[2]), int(tokens[4]))
                upLeftY = min(int(tokens[3]), int(tokens[5]))
                newEntry += " 1 " + str(upLeftX) + " " + str(upLeftY)

                width = abs(int(tokens[2]) - int(tokens[4]))
                height = abs(int(tokens[3]) - int(tokens[5]))

                if upLeftX + width > 640:
                    width = 640 - upLeftX
                    print upLeftX + width
                if upLeftY + height > 480:
                    height = 480 - upLeftY
                    print upLeftY + height

                newEntry += " " + str(width) + " " + str(height) + "\n"
                convertedData.append(newEntry)

        if not imgFolder:
            imgFolder = self.params["ImagePath"][0]

        destinationName = os.path.join(imgFolder, newAnnotationsName)
        with open(destinationName, "w") as outputFile:
            for data in convertedData:
                outputFile.write(data)

        return destinationName

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

        positiveCollection, negativeCollection = self.parseImagePaths()

        if isinstance(self.params["AnnotationFiles"], list):
            annotationsPath = self.params["AnnotationFiles"][0]
        else:
            annotationsPath = self.params["AnnotationFiles"]
        if isinstance(self.params["width"], list):
            width = self.params["width"][0]
        else:
            width = self.params["width"]
        if isinstance(self.params["height"], list):
            height = self.params["height"][0]
        else:
            height = self.params["height"]

        completionFlag = False

        # Create, if necessary, the directory where the dataset will
        # be saved.
        samplesDestinationDir = os.path.relpath(self.dataSetFolder)

        if (not os.path.isdir(samplesDestinationDir)):
            os.mkdir(samplesDestinationDir)

        # Gather the paths to all the positive samples in a single txt
        # file.
        positiveSamples = os.path.join(samplesDestinationDir, "positives.txt")
        with open(positiveSamples, "w") as f:
            for image in positiveCollection["training"]:
                f.write(os.path.abspath(image) + "\n")

        # Gather the paths to all the negative samples in a single txt
        # file.
        negativeSamples = os.path.join(samplesDestinationDir, "negatives.txt")
        with open(negativeSamples, "w") as f:
            for image in negativeCollection["training"]:
                f.write(os.path.abspath(image) + "\n")

        validationData = os.path.join(samplesDestinationDir, "Validation.txt")
        validationSet = positiveCollection["test"] + negativeCollection["test"]
        with open(validationData, "w") as f:
            for entry in validationSet:
                f.write(os.path.abspath(entry.strip(" \n\t")) + "\n")

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
            samplesDestinationDir = os.path.join(samplesDestinationDir,
                                                 "samples.txt")
            with open(samplesDestinationDir, "w") as samplesFile:
                # Iterate over all the files in the samples folder
                # and copy the path of all the ".vec" files in
                # the ".txt" file.
                vecDir = os.path.join(samplesDestinationDir, "vec")
                for file in os.listdir(vecDir):
                    file = os.path.join(vecDir, file)
                    if (os.path.isfile(file) and ".vec" in file):
                        # Write the path to the file.
                        samplesFile.write(vecDir + "/" + file + "\n")

            # Merge the samples in one ".vec" file.

            samplesOriginPath = os.path.join(samplesDestinationDir,
                                             "samples.txt")
            mergeFlag = self.mergeSamples(samplesOriginPath,
                                          samplesDestinationDir)
            completionFlag = completionFlag or mergeFlag
        else:
            newAnnotationsName = "info.dat"
            newAnnotationsPath = self.convertAnnotations(annotationsPath,
                                                         newAnnotationsName)
            vecCreation = self.createTrainingSamples(samplesDestinationDir,
                                                     newAnnotationsPath,
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
                # If there is only one element then don't create a list but
                # just assign the value to the dictionary entry.
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
            input = raw_input("Please enter an email address (or more) to" +
                              " notify" + " you when the training procedure" +
                              " completes: ")

            # Write it to the configuration file.
            utils.writeCSV(config, "MailDestination", input)

            input = raw_input("Do you want to use annotated data? [Y/N] ")
            if "Y" in input:
                input = raw_input("Please enter the path to the annotation's" +
                                  " file: ")
                input = input.split(" ")
                utils.writeCSV(config, "AnnotationFiles", input)

                input = raw_input("Please enter the path to the folder" +
                                  " that contains the images : ")
                utils.writeCSV(config, "ImagePath", input)
            else:
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

                utils.writeCSV(config, "Positive Directory", posDir)

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
