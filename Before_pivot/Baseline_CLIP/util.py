import os
import json
from PIL import Image
import re
import shutil

def save_as_json(content, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    print(f"Saved content to {path}")

def load_json(path):
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded content from {path}")
    return data

def detect_file(file_path):
    if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
    
def show_image(image_path):
    img = Image.open(image_path)
    img.show() 

def sort_key(filename):
    pattern = r"Page(\d+)\.txt$"
    match = re.match(pattern, filename)
    if match:
        page_num = int(match.group(1))
        return page_num
    
def read_query(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        query_text = f.read().strip()
    return query_text

def clear_output_folder(folder_path):
    """
    If the folder exists then clear all its content.
    Otherwise create a new folder.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"{folder_path} is cleared!")
    os.makedirs(folder_path, exist_ok=True)
    print(f"{folder_path} is ready for new content.")

def create_class_from_json(cls, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    instances = [cls.from_dict(data) for data in data_list]
    return instances