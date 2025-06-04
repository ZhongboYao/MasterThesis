import os
import json
import shutil
import re
import random

def save_as_json(content: list, path: str):
    """
    Save the content as a .json file stored at path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    print(f"Saved content to {path}")

def load_json(path: str) -> list:
    """
    Load content from the .json file.
    """
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded content from {path}")
    return data

def clear_folder(folder_path: str) -> None:
    """
    If the folder exists then clear all its content.
    Otherwise create a new folder.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"{folder_path} is cleared!")
    os.makedirs(folder_path, exist_ok=True)
    print(f"{folder_path} is ready for new content.")

def save_as_txt(text: str, file_path: str):
    """
    Save the content to the .txt file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    
def read_query(file_path: str) -> str:
    """
    Read query from a .txt file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        query_text = f.read().strip()
    return query_text

def random_answer(original_answer, answer_list):
    new_ans = original_answer
    while new_ans == original_answer:
        new_ans = random.choice(answer_list)
    return new_ans
    
