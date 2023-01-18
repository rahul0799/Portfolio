"""This module is used to raise customs exceptions."""


class CallingUtilititesClass(Exception):
    """Raised when Utilities Class is not called"""

    def __str__(self):
        return "Exception in calling Utilities Class."


class CallingConfigParser(Exception):
    """Raised when ConfigParser Class is not called"""

    def __str__(self):
        return "Exception in calling CallingConfigParser Class."


class ErrorResizing(Exception):
    """Raised when exception in resizing"""

    def __str__(self):
        return "Exception is raised when exception in resizing."


class ErrorColor(Exception):
    """Raised when exception in changing colour"""

    def __str__(self):
        return "Exception is raised when exception in changing colour."


class ErrorConvertingToPIL(Exception):
    """Raised when exception in converting numpy img to pil"""

    def __str__(self):
        return "Exception is raised when exception in converting numpy img to pil."


class ErrorConvertingToNumpy(Exception):
    """Raised when exception in converting pil img to numpy"""

    def __str__(self):
        return "Exception is raised when exception in converting pil img to numpy."


class InfiniteLoop(Exception):
    """Raised when loop goes to infinity"""

    def __str__(self):
        return "Exception is raised when loop goes to infinity."


class ErrorRemovingBox(Exception):
    """Raised when error in removing box"""

    def __str__(self):
        return "Exception is raised when error in removing box."


class ErrorRemovingColour(Exception):
    """Raised when error in removing colour"""

    def __str__(self):
        return "Exception is raised when error in removing colour."


class HeightError(Exception):
    """Raised when error in calculating average height of box"""

    def __str__(self):
        return "Exception is raised when error in calculating average height of box."
