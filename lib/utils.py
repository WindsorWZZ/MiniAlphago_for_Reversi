import os

def read_as_int(filename):
    """
    read file int
    :param filename:
    :return:
    """
    if os.path.exists(filename):
        with open(filename, "rt") as f:
            return int(str(f.read()).strip())


