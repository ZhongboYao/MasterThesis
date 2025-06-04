import util
import evaluation
import pandas as pd
from typing import Tuple
from openai import OpenAI
import api    
import json
import re

def extract_annotations(doc_ann_path: str, gpt_ann_path: str, result_path: str) -> Tuple[list, list, list]:
    """
    The function extracts doctor's and gpt's annotations from a .json file.
    Then the doctor and gpt's annotations are saved together to a .xlsx file for better comparison.
    
    Parameters
    -----------
    doc_ann_path, gpt_ann_path: str
        Paths contaning the two annotations.
    result_path: str
        The path to save the doctor and gpt annotations together as a comparison.
        Should be .xlsx.
    
    Returns
    --------
    doc_anns: List[list], #Cases * #Questions
        A list containing all the cases, each case is a list contatning all the answers to 14 questions
    gpt_anns: List[list], #Cases * #Quesions
        Similar to doc_anns.
    features: list, #Questions
        The names of questions.
    """
    doc = util.load_json(doc_ann_path)
    gpt = util.load_json(gpt_ann_path)

    features = list(doc[0].keys())[4:]
    result_index = features + ['acc']
    result = pd.DataFrame(index=result_index)

    doc_anns = []
    gpt_anns = []

    # Iterate cases.
    for i, (doc_ann, gpt_ann) in enumerate(zip(doc, gpt), start=1):
        doc_answers = list(doc_ann.values())[4:]
        gpt_answers = list(json.loads(gpt_ann).values())

        acc = evaluation.exact_match(doc_answers, gpt_answers)
        doc_answers.append('-')
        gpt_answers.append(acc)

        result[f'doc_case_{i}'] = doc_answers
        result[f'gpt_case_{i}'] = gpt_answers

        doc_anns.append(doc_answers)
        gpt_anns.append(gpt_answers)

    result.to_excel(result_path)
    return doc_anns, gpt_anns, features

def inconsistency_orgnize_annotations(doc_ann_path: str, gpt_ann_path: str, fake_ann_path: str, result_path: str) -> tuple:
    """
    The special version of organize_annotations for inconsistency check.
    This one needs to read doctor's, gpt's annotations and the annotation with errors.
    """
    extract_contents = util.load_json(doc_ann_path)
    annotations = util.load_json(gpt_ann_path)
    fakes = util.load_json(fake_ann_path)

    features = list(extract_contents[0].keys())[4:]
    result_index = features + ['acc']
    result = pd.DataFrame(index=result_index)

    doc_anns = []
    gpt_anns = []
    fake_anns = []

    for i, (doc_ann, gpt_ann, fake_ann) in enumerate(zip(extract_contents, annotations, fakes), start=1):
        doc_answers = list(doc_ann.values())[4:]
        fake_answers = list(fake_ann.values())[4:]
        gpt_answers = gpt_ann.split('- ')[1:]
        gpt_answers = [item.replace('\n', '').replace('Answer: ', '').strip() for item in gpt_answers]

        acc = evaluation.exact_match(doc_answers, gpt_answers)
        doc_answers.append('-')
        fake_answers.append('-')
        gpt_answers.append(acc)
        
        result[f'doc_case_{i}'] = doc_answers
        result[f'gpt_case_{i}'] = gpt_answers
        result[f'fake_case_{i}'] = fake_answers

        doc_anns.append(doc_answers)
        fake_anns.append(fake_answers)
        gpt_anns.append(gpt_answers)

    result.to_excel(result_path)
    return doc_anns, gpt_anns, fake_anns, features


def extract_answers(questions: str, query: str) -> str:
    """
    This function extracts information from messy GPT output to .json file.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    messages = [
        {
            "role": "system",
            "content": (
                f"""
                You are an expert in information extraction. 
                Your task is to generate a json object, with keys the questions and values the extracted corresponding answers.
                Output only valid JSON with no Markdown formatting or triple backticks.
                Below are the 14 questions and the allowed answers you are going to consider, except the final quesiton which is about diagnosis.
                Pick exactly one answer for the first 13 questions.
                For the final (14th) question, provide your own free-form response.
                {questions}
                """
            )
        },
        {
            "role": "system",
            "content": (
                """
                Use the following structure for each item in the json file:
                - <Nr>. <Question>: <Answer>

                For example:
                - 12. Does the lesion result in root resorption?: No",

                There are 14 questions, do not miss anyone.
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"""
                        Below is the text you are going to use to extract answers and store as a json file.
                        {query}
                        """
                    )   
                }
            ]
        }
    ]
    
    annotation = []
    while not annotation:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content
    return annotation


def extract_answers_specific_question(questions: str, information: str, to_answer: str) -> str:
    """
    This function extracts information from messy GPT output to .json file.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    messages = [
        {
            "role": "system",
            "content": (
                f"""
                You are an expert in information extraction. 
                Your task is to generate a json object, with keys the questions and values the extracted corresponding answers.
                Output only valid JSON with no Markdown formatting or triple backticks.
                Below are all the possible questions and the allowed answers you are going to consider.
                {questions}
                But you are just going to extract answers for some of them based on the query, from the given information.
                """
            )
        },
        {
            "role": "system",
            "content": (
                """
                Use the following structure for each item in the json file:
                - <Question>: <Answer>

                For example:
                - Does the lesion result in root resorption?: No,
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"""
                        Below is the text you are going to use to extract the answer to the question and store as a json file.
                        {information}
                        """
                    )   
                },
                {
                    "type": "text",
                    "text": (
                        f"""
                        Below are the questions for extracting answers.
                        {to_answer}
                        """
                    )   
                }
            ]
        }
    ]
    
    annotation = []
    while not annotation:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
        )
        annotation = response.choices[0].message.content
    return annotation

def process_response(response):
    # Ensure response is a string
    if not isinstance(response, str):
        raise ValueError(f"Expected string, got {type(response)}")

    # Clean the response
    cleaned_response = response.strip()

    # Handle triple-backtick JSON format (e.g., ```json ... ```)
    if cleaned_response.startswith('```json'):
        # Extract content between ```json and ```
        match = re.search(r'```json\s*(.*?)\s*```', cleaned_response, re.DOTALL)
        if match:
            cleaned_response = match.group(1).strip()
        else:
            raise ValueError("Invalid triple-backtick JSON format")

    # Remove single quotes surrounding JSON (if present)
    cleaned_response = cleaned_response.strip("'").strip('"')

    # Try parsing as JSON
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}, input: {cleaned_response}")
    
def question_locate(questions: str, information: str) -> str:
    """
    This function extracts information from messy GPT output to .json file.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    messages = [
        {
            "role": "system",
            "content": (
                f"""
                You are an expert in information extraction. 
                The given information is from a doctor, which is a question - answer pair.
                Your task is to select the question contained in the given doctor's information for the following list:
                {questions}
                Output only valid string.
                """
            )
        },
        {
            "role": "system",
            "content": (
                """
                Use the following structure for the output string:
                String

                For example:
                Does the lesion result in root resorption?

                You should pick the question exactly from the given list without any modifications.
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"""
                        Below is the text you are going to use to extract the question.
                        {information}
                        """
                    )   
                }
            ]
        }
    ]
    
    annotation = []
    while not annotation:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
        )
        annotation = response.choices[0].message.content
    return annotation