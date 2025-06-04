import fitz
from tqdm import tqdm
import Vision_Model_Exploration.util as util
import base64
from pdf2image import convert_from_path
from PIL import Image as Img
import Vision_Model_Exploration.api as api
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

openai_client = OpenAI(api_key=api.OPENAI_KEY)

def get_texts_from_pdf_page(pdf_page):
    text = pdf_page.get_text()
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    cleaned_text = '\n'.join(non_empty_lines)
    return cleaned_text

def get_images_from_pdf_page(doc, pdf_page):
    extracted_images = []
    image_list = pdf_page.get_images(full=True)

    if image_list:
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image(image_bytes, image_ext)
            extracted_images.append(image)

    return extracted_images

def create_queries(para, method):
    model_name = 'doc2query/msmarco-14langs-mt5-base-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(para, return_tensors='pt')
    with torch.no_grad():
        if method == 1:
            outputs = model.generate(
                input_ids=input_ids,
                max_length=64,
                do_sample=True,
                top_p=0.95,
                top_k=10, 
                num_return_sequences=5
                )
        
        else:
            outputs = model.generate(
                input_ids=input_ids, 
                max_length=64, 
                num_beams=5, 
                no_repeat_ngram_size=2, 
                num_return_sequences=5, 
                early_stopping=True
            )
        
        query = []
        for i in range(len(outputs)):
            query.append(tokenizer.decode(outputs[i], skip_special_tokens=True))
        return query
        
class Image:
    def __init__(self, bytes=None, ext=None, index=None, parent_page_num=None, parent_file_path=None, path=None, caption=None, context_page_nums=[]):
        self.bytes = bytes
        self.ext = ext
        self.index = index
        self.parent_page_num = parent_page_num
        self.parent_file_path = parent_file_path
        self.path = path
        self.caption = caption
        self.context_page_nums = context_page_nums

    def save(self, save_path):
        image_path = f"{save_path}"
        with open(image_path, "wb") as img_file:
            img_file.write(self.bytes)

    def pair_caption(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as caption_file:
            self.caption = caption_file.read()
    
    def get_pages_context(self):
        doc = fitz.open(self.parent_file_path)
        max_num = len(doc)-1
        current_page_num = self.parent_page_num
        previous_page_num = max(0, current_page_num-1)
        afterwards_page_num = min(max_num, current_page_num+1)
        self.context_page_nums = [previous_page_num, current_page_num, afterwards_page_num]

    def generate_caption(self, img_context, txt_context):
        with open(self.path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        txt_context = txt_context

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
                            "Give the symptom of the image and extract all its description from the given text context. You should only use the provided text, and as detailed as possible."
                            "Use the structure of Symptom: xxx, Description: xxx."
                        )
                    },
                    # First context image
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_context[0]}"
                        }
                    },
                    # Second context image
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_context[1]}"
                        }
                    },
                    # Third context image
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_context[2]}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Here are texts extracted from pdf images above: {txt_context}"
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
        filtered_data = {k: v for k, v in data.items() if k not in ("bytes", "ext")}
        return cls(**filtered_data)
    
class Page:
    def __init__(self, page, parent_file_path, page_num):
        self.parent_file_path = parent_file_path
        self.page_num = page_num
        self.text = None
        self.images = []
        self.page = page

    def process_images(self):
        for i in range(len(self.images)):
            image = self.images[i]
            image.index = i
            image.parent_page_num = self.page_num
            image.parent_file_path = self.parent_file_path

    def convert_to_image(self):
        pix = self.page.get_pixmap()
        img = Img.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

class PDF:
    def __init__(self, path):
        self.path = path
        self.name = self._get_name()
        self.doc = fitz.open(self.path)
        self.pages = []
        self.images = []

    def _get_name(self):
        relative_path_name = self.path.split('/')[-1]
        name = relative_path_name.split('.')[0]
        return name
    
    def extract_pages(self):
        for page_num in tqdm(range(len(self.doc)), desc=f"Processing pages of file {self.path}"):
            pdf_page = self.doc.load_page(page_num)
            page = Page(pdf_page, self.path, page_num)

            # Get texts
            page.text = get_texts_from_pdf_page(pdf_page)
            # Get images
            page.images = get_images_from_pdf_page(self.doc, pdf_page)
            page.process_images()
            self.images.extend(page.images)

            self.pages.append(page)

    def combine_pages_txt(self):
        content = ""
        for page in self.pages:
            content += page.text
        return content
    
    def save_images_info(self, save_path):
        excluded_keys = {'bytes', 'ext'}
        images_dict = [{k: v for k, v in image.__dict__.items() if k not in excluded_keys} for image in self.images]
        util.save_as_json(images_dict, save_path)
    
    