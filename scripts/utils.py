#! /usr/bin/python

import os

BOLD = "\033[1m"
GREEN = "\x1b[032;1m"
BLUE = "\x1b[034;1m"
RED = "\x1b[031;1m"
YELLOW = "\x1b[033;1m"
ENDC = "\033[0m"
BRED = BOLD + RED
BGREEN = BOLD + GREEN
BBLUE = BOLD + BLUE
BYELLOW = BOLD + YELLOW


class Rectangle:

    """A class that represents a 2D rectangle"""

    def __init__(self, points):
        """The constructor that takes as input 2 points that represent the
           top left and bottom right corner of a rectangle and create the
           corresponding object.

        @param points(tuple): x, y coordinates of the top left and bottom
                              right corner.
        """
        # Convert the tuple elements to integers
        if isinstance(points, tuple) or isinstance(points, list):
            points = [int(x) for x in points]
        self.x = min(points[0], points[2])
        self.y = min(points[1], points[3])

        self.width = max(points[0], points[2]) - self.x
        self.height = max(points[1], points[3]) - self.y

    def calc_area(self):
        """Calculates the area of the rectangle """
        return self.width * self.height


def value_in_range(val, minVal, maxVal):
    return (val >= minVal) and (val <= maxVal)


def check_overlap(A, B):
    """Function that checks whether two rectangles overlap.

    @param A(Rectangle): The first rectangle
    @param B(Rectangle): The second rectangle
    @return: True if the the rectangles overlap more than 10 percent.

    """
    # Check if the first argument is a rectangle.
    if not isinstance(A, Rectangle):
        A = Rectangle(A)
    # Check if the second argument is a rectangle.
    if not isinstance(B, Rectangle):
        B = Rectangle(B)

    # Check if the x coordinate of the top left coordinate
    # of each rectangle is within the range defined by the other.

    # Check if the x coordinate of the first rectangle is in the second one.
    x_a_in_b = value_in_range(A.x, B.x, B.x + B.width)
    # Check if the x coordinate of the second rectangle is in the first one.
    x_b_in_a = value_in_range(B.x, A.x, A.x + A.width)
    xOverlap = x_a_in_b or x_b_in_a

    # Check if the y coordinate of the first rectangle is in the second one.
    y_a_in_b = value_in_range(A.y, B.y, B.y + B.height)
    # Check if the y coordinate of the second rectangle is in the first one.
    y_b_in_a = value_in_range(B.y, A.y, A.y + A.height)
    yOverlap = y_a_in_b or y_b_in_a

    # Check if the two rectangles overlap.
    if xOverlap and yOverlap:
        # If yes find the common surface defined by the two of them.
        topLeftCommon = None
        bottomRightCommon = (min(A.x + A.width, B.x + B.width),
                             min(A.y + A.height, B.y + B.height))

        if x_a_in_b and y_a_in_b:
            topLeftCommon = (A.x, A.y)
        elif x_b_in_a and y_b_in_a:
            topLeftCommon = (B.x, B.y)
        elif x_b_in_a and y_a_in_b:
            topLeftCommon = (B.x, A.y)
        elif x_a_in_b and y_b_in_a:
            topLeftCommon = (A.x, B.y)

        overlappingRectangle = ([point for point in topLeftCommon] +
                                [point for point in bottomRightCommon])
        overlappingRectangle = Rectangle(overlappingRectangle)
        # Check if their common surface is more than 10 percent.
        if overlappingRectangle.calc_area() / float(B.calc_area()) >= 0.10:
            return True
        else:
            return False
    else:
        return False


def fileSearch(fileName):
    """Function used to search for the binary file given by the argument.

    @param fileName(str): The name of the file we want to find.
    @return: Returns a string that is the path to the file.

    """
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


def writeCSV(file, tag, data):
    """Write the input data as comma separated values with a description tag
       to the provided file with a

    @param file(file): The file where the data will be written.
    @param tag(str): The tag/id of the data.
    @param data (str): The data we want to write to the file.
    @return: None

    """
    if (isinstance(data, list)):
        file.write(tag + " : " + ",".join(data) + " \n")
    else:
        data = data.split()
        file.write(tag + " : " + ",".join(data) + " \n")
