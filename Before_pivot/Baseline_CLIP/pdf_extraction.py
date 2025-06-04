import os
import shutil
import fitz
from pdf2image import convert_from_path
from tqdm import tqdm
import Vision_Model_Exploration.util as util
import json
import base64
from openai import OpenAI
import Vision_Model_Exploration.api as api

openai_client = OpenAI(api_key=api.OPENAI_KEY)

class Image:
    def __init__(self, image_path, pdf_path, page_num, image_index, caption="", context_page_nums=[]):
        self.image_path = image_path
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.image_index = image_index
        self.caption = caption
        self.context_page_nums = context_page_nums
    
    def pair_caption(self, singlepage_folder):
        pdf_name = self.pdf_path.split('/')[-1]
        pdf_name = pdf_name.split('.')[0]

        corr_txt_path = f"{singlepage_folder}/{pdf_name}/Page{self.page_num}.txt"
        with open(corr_txt_path, 'r', encoding='utf-8') as caption_file:
            self.caption = caption_file.read()
    
    def generate_caption(self, singlepage_folder):
        pdf_name = self.pdf_path.split('/')[-1]
        pdf_name = pdf_name.split('.')[0]

        with open(self.image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        base64_page_images = []
        pdf_context = ""
        for i in range(len(self.context_page_nums)):
            page_image = f"{singlepage_folder}/{pdf_name}/Page{self.context_page_nums[i]}.png"
            with open(page_image, "rb") as image_file:
                base64_page_image = base64.b64encode(image_file.read()).decode('utf-8')
                base64_page_images.append(base64_page_image)

            pdf_txt = f"{singlepage_folder}/{pdf_name}/Page{self.context_page_nums[i]}.txt"
            with open(pdf_txt, "r") as text_file:
                file_text = text_file.read()
                pdf_context += file_text + "\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an image reading expert. Summarize the description "
                    "of the last image given the first three images as context."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Below are three pdf images providing context, followed by a final image extracted from one of these pdfs."
                            "Please analyze the three context pdf images, then extract the symptom name and description of the image."
                            "The text content from the three pdf images are also provided below for your better understanding."
                            "Give the symptom of the image and a very detailed description of it."
                            "Use the structure of Symptom: xxx, Description: xxx."
                        )
                    },
                    # First context image
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_page_images[0]}"
                        }
                    },
                    # Second context image
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_page_images[1]}"
                        }
                    },
                    # Third context image
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_page_images[2]}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Here are texts extracted from pdf images above: {pdf_content}"
                        )
                    },
                    {
                        "type": "text",
                        "text": (
                            "Now here is the final image to summarize, only use texts from the given PDF files:"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
            
        response = openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            max_tokens=300,
        )
        self.caption = response.choices[0].message.content

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class PDF:
    def __init__(self, file_path):
        self.file_path = file_path
        relative_path_name = self.file_path.split('/')[-1]
        self.file_name = relative_path_name.split('.')[0]
        self.txt_pages = {}
        self.image_index = 0
        self.images = []
        self.txt_content = ""

    def convert_to_images(self, singlepage_folder):
        print('Converting pdf pages to individual images.')
        pages = convert_from_path(self.file_path, dpi=200)
        for j, page in tqdm(enumerate(pages)):
            page.save(f"{singlepage_folder}/{self.file_name}/Page{j+1}.png", "PNG")

    def extract_singlepage_text(self, page, page_num, singlepage_folder):
        text = page.get_text()
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        cleaned_text = '\n'.join(non_empty_lines)

        os.makedirs(f"{singlepage_folder}/{self.file_name}", exist_ok=True)
        single_page = f"{singlepage_folder}/{self.file_name}/Page{page_num+1}.txt"
        with open(single_page, "a", encoding="utf-8") as single_txt:
            single_txt.write(cleaned_text)
        
        self.txt_pages[page_num] = cleaned_text

    def extract_images(self, doc, page, page_num, extract_folder):
        image_list = page.get_images(full=True)

        if image_list:
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                os.makedirs(f"{extract_folder}/{self.file_name}/Images", exist_ok=True)
                image_path = f"{extract_folder}/{self.file_name}/Images/Figure{self.image_index+1}.{image_ext}"
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                image = Image(image_path, self.file_path, page_num+1, self.image_index+1)
                self.images.append(image)

                self.image_index += 1

    def combine_text(self, source_folder, extract_folder):
        text_files = [f for f in os.listdir(source_folder) if f.endswith(".txt")]
        text_files.sort(key=util.sort_key)
        
        os.makedirs(f"{extract_folder}/{self.file_name}", exist_ok=True)
        output_path = f"{extract_folder}/{self.file_name}/{self.file_name}_Pages.txt"
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for txt_file in text_files:
                file_path = os.path.join(source_folder, txt_file)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    contents = infile.read()
                    outfile.write(contents)
                    outfile.write("\n")
                self.txt_content += contents
                self.txt_content += "\n"

        print("Text files are combined in:", output_path)

    def extract_images_and_text(self, singlepage_folder, extract_folder):
        os.makedirs(singlepage_folder, exist_ok=True)

        doc = fitz.open(self.file_path)
        
        for page_num in tqdm(range(len(doc)), desc=f"Processing pages of file {self.file_path}"):
            page = doc.load_page(page_num)
            self.extract_singlepage_text(page, page_num, singlepage_folder)
            self.extract_images(doc, page, page_num, extract_folder)

        self.combine_text(f"{singlepage_folder}/{self.file_name}", extract_folder)

    def save_imageInfo(self, extract_folder):
        data = [image.__dict__ for image in self.images]
        os.makedirs(f"{extract_folder}", exist_ok=True)
        with open(f"{extract_folder}/{self.file_name}/{self.file_name}_imagesInfo.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Data written to {extract_folder}/{self.file_name}/{self.file_name}_imagesInfo.json")

    def append_images_contexts(self):
        doc = fitz.open(self.file_path)
        max_num = len(doc)
        for image in self.images:
            current_page_num = image.page_num
            previous_page_num = max(1, current_page_num-1)
            afterwards_page_num = min(max_num, current_page_num+1)
            image.context_page_nums.extend([previous_page_num, current_page_num, afterwards_page_num])
