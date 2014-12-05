import os

def get_samples_dir():
    return os.path.join(os.path.dirname(__file__), 'samples')

def get_samples_file(filename):
    return os.path.join(get_samples_dir(), filename)