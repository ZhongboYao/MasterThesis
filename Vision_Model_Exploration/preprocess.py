import re
import glob
from typing import List
import random
import util

def extract_number(filename: str):
    """
    Extract the indices from file names.
    """
    match = re.search(r'case(\d+)\.png', filename)
    return int(match.group(1)) if match else 0

def get_all_image_paths(filename_template: str = 'evaluation_dataset/images/case*.png') -> List[str]:
    """
    Given the template of image filenames (including the folder name), the function returns 
    a list containing all the associative image file paths. Filenames are also sorted in 
    the returned list according to their indices in the names.

    Parameters
    ----------
    filename_template: str
        The filename structure.
    
    Returns
    -------
    image_filenames: List[str]
        A list containing all the images names.
    """
    image_filenames = glob.glob(filename_template)
    image_filenames.sort(key=extract_number)
    return image_filenames

def extract_questions(path: str) -> List[str]:
    """
    Get questions from a .json file and store them in a list.
    """
    questions = util.load_json(path)['questions']
    return questions

def randomized_list(original_list: list, n: int) -> list:
    """
    Shuffle the first n elements of the original list.
    """
    temp_list = original_list[:]
    prefix = temp_list[:n]
    random.shuffle(prefix)
    shuffled = prefix + temp_list[n:]
    return shuffled
