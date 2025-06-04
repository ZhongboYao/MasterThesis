import base64
from openai import OpenAI
import os
from tqdm import tqdm
import Vision_Model_Exploration.api as api

openai_client = OpenAI(api_key=api.OPENAI_KEY)

class CaptionGenerator:

    def __init__(self, extract_folder, pdf_img_folder):
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
        self.extract_folder = extract_folder
        self.pdf_img_folder = pdf_img_folder
        self.caption_pairs = []

    def pair_captions(self, image_contexts):
        print("Paring captions for images.")
        for i in tqdm(range(len(image_contexts))):
            caption_path = os.path.join(self.pdf_img_folder, image_contexts[i][2][0])
            image_path = os.path.join(self.extract_folder, image_contexts[i][0])
            with open(caption_path, 'r', encoding='utf-8') as file:
                caption = file.read()
            self.caption_pairs.append((caption, image_path))


