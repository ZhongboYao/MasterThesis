import Vision_Model_Exploration.util as util
import os
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI
import Vision_Model_Exploration.api as api
from fastembed import SparseTextEmbedding

openai_client = OpenAI(api_key=api.OPENAI_KEY)

class Chunk:
    def __init__(self, content):
        self.content = content 

def naive_chunk(chunk_size, overlap, text_file_path):
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        chunks = text_splitter.split_text(text)
        print(f"Total text chunks created: {len(chunks)}")
        return chunks

def single_page_chunk(pages_folder):
    chunks = []

    text_files = [f for f in os.listdir(pages_folder) if f.endswith(".txt")]
    text_files.sort(key=util.sort_key)
    
    for txt_file in text_files:
        file_path = os.path.join(pages_folder, txt_file)
        with open(file_path, 'r', encoding='utf-8') as infile:
            contents = infile.read().strip()
            chunks.append(contents)
    
    return chunks