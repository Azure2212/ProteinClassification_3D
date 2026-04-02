import os

def get_classes(path):
    allProteins = os.listdir(path)
    classes = {}
    for idx, name in enumerate(allProteins):
        classes[idx] = name
    return classes