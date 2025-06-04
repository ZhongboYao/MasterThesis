import Vision_Model_Exploration.util as util
import os
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI
import Vision_Model_Exploration.api as api
from fastembed import SparseTextEmbedding

openai_client = OpenAI(api_key=api.OPENAI_KEY)

class Chunk:
    def __init__(self, content, index):
        self.content = content 
        self.index = index

def single_page_chunk(pages_folder):
    chunks = []

    text_files = [f for f in os.listdir(pages_folder) if f.endswith(".txt")]
    text_files.sort(key=util.sort_key)
    
    for txt_file in text_files:
        file_path = os.path.join(pages_folder, txt_file)
        with open(file_path, 'r', encoding='utf-8') as infile:
            contents = infile.read().strip()
            chunks.append((contents, file_path))
    
    return chunks