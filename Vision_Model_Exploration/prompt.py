import base64
from openai import OpenAI
import api    
import re
import heapq
from typing import List, Tuple
from tqdm import tqdm
import anthropic

refusal_pattern = re.compile(
    r"\bI['’]m unable to\b",
    flags=re.IGNORECASE
)

def zeroshot(image_path: str, questions: str) -> str:
    """
    Generate radiographic annotations with ChatGPT-4o Vision.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    # Load and inline‐encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                "You are an oral radiology expert assistant. "
                "Analyze the oral panoramic image thoroughly and provide your internal reasoning in free‐form. "
                "Then, for each numbered question, give a concise final answer drawn from the listed options. "
                "Do not say “I can’t analyze images.”"
                "This is for exploratory/educational use only."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the oral panoramic image to annotate:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Answer each question by number, selecting from the provided choices."
                        "For the final question, provide your own free-form response."
                        f"{questions}\n\n"
                        "Finally, list each question number followed by your answer."
                    )
                }
            ]
        }
    ]

    # Retry until non‐empty
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

def zeroshot_explained(image_path: str, questions: str) -> str:
    """
    Generate radiographic annotations with ChatGPT-4o Vision.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    # Load and inline‐encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                "You are an oral radiology expert assistant. "
                "Analyze the oral panoramic image thoroughly and provide your internal reasoning in free‐form. "
                "Then, for each numbered question, give a concise final answer drawn from the listed options. "
                "Do not say “I can’t analyze images.”"
                "This is for exploratory/educational use only."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the oral panoramic image to annotate:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Answer each question by number, selecting from the provided choices."
                        "For some questions, an explain of the associated choice is also provided for you better understanding."
                        "For the final question, provide your own free-form response."
                        f"{questions}\n\n"
                        "Finally, list each question number followed by your answer."
                    )
                }
            ]
        }
    ]

    # Retry until non‐empty
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

def zeroshot_free(image_path: str, questions: str) -> str:
    """
    Generate radiographic annotations with ChatGPT-4o Vision.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    # Load and inline‐encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                "You are an oral radiology expert assistant. "
                "Analyze the oral panoramic image thoroughly and provide your internal reasoning in free‐form. "
                "Then, for each numbered question, give a concise final answer drawn from the listed options. "
                "Do not say “I can’t analyze images.”"
                "This is for exploratory/educational use only."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the oral panoramic image to annotate:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Answer each question by number."
                        "For the first 13 questions, there are also some answers for reference."
                        "You can generate your responses in free-form."
                        f"{questions}\n\n"
                        "Finally, list each question number followed by your answer."
                    )
                }
            ]
        }
    ]

    # Retry until non‐empty
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

def fewshots(image_path: str, questions: str, examples: str) -> str:
    """
    Prompts with few shots.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                "You are an oral radiology expert assistant. "
                "Analyze the oral panoramic image thoroughly and provide your internal reasoning in free‐form. "
                "Then, for each numbered question, give a concise final answer drawn from the listed options. "
                "Do not say “I can’t analyze images.”"
                "This is for exploratory/educational use only."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below are two sets of example answers with the corresponding questions for you to generate more accurate responses.
                        """
                    )
                },
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {examples}
                """
            )   
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below is the image that you are going to help annotate.
                        """
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                        f"""
                        "Answer each question by number, selecting from the provided choices."
                        "For the final question, provide your own free-form response."
                        f"{questions}\n\n"
                        "Finally, list each question number followed by your answer."
                        """
                    )   
                }
            ]
        }
    ]
    
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

def fewshotsImg(image_path: str, questions: str, example_img1_path: str, example1:str, example_img2_path: str, example2: str) -> str:
    """
    Prompts with fewshots with the corresponding images.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    with open(example_img1_path, "rb") as image_exp1:
        img64_exp1 = base64.b64encode(image_exp1.read()).decode('utf-8')
    with open(example_img2_path, "rb") as image_exp2:
        img64_exp2 = base64.b64encode(image_exp2.read()).decode('utf-8')
        
    messages = [
        {
            "role": "system",
            "content": (
                "You are an oral radiology expert assistant. "
                "Analyze the oral panoramic image thoroughly and provide your internal reasoning in free‐form. "
                "Then, for each numbered question, give a concise final answer drawn from the listed options. "
                "Do not say “I can’t analyze images.”"
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below are two sets of example answers with the corresponding questions for you to generate more accurate responses.
                        """
                    )
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img64_exp1}"
                    }
                }
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {example1}
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img64_exp2}"
                    }
                }
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {example2}
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below is the image that you are going to help annotate.
                        """
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                        f"""
                        "Answer each question by number, selecting from the provided choices."
                        "For the final question, provide your own free-form response."
                        f"{questions}\n\n"
                        "Finally, list each question number followed by your answer."
                        """
                    )   
                }
            ]
        }
    ]
    
    annotation = ""
    while not annotation.strip():
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content
    return annotation

def CoTFewshots(image_path: str, questions: str, examples: str, temperature=0.0) -> str:
    """
    Prompts with chain of thoughts.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant in annotating panoramic image of oral lesions. "
                "Use a chain‐of‐thought approach: think step by step before giving each final answer. "
                "Do not refuse by saying 'I can’t analyze images.' "
                "This is for exploratory/educational use only."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below are two sets of example answers with the corresponding questions for you to generate more accurate responses.
                        """
                    )
                },
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {examples}
                """
            )   
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below is the image that you are going to help annotate.
                        """
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Let’s think step by step.\n\n"
                        "For each of the following questions, provide your internal reasoning followed by "
                        "a final answer (choose from the provided options for Q1–Q13, free-form for Q14):\n\n"
                        f"{questions}\n\n"
                        "At the end, list each question number and your answer."
                    )   
                }
            ]
        }
    ]
    
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=temperature,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

def score_branch(branch_text: str, image_b64, question) -> float:
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant in annotating panoramic image of oral lesions. "
                "You are going to judge how confident an answer to a question is regarding to an image."
                "Do not refuse by saying 'I can’t analyze images.' "
                "This is for exploratory/educational use only."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below is the image which the question and answer is about.
                        """
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"""
                        Below is the question and the associated reasoning and answer.
                        {question}
                        {branch_text}
                        On a scale from 0.0 (no confidence) to 1.0 (maximum confidence), how confident are you that you think it is correct?
                        Please reply with a single floating-point number.
                        """
                    )
                },
            ]
        }
    ]
    
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    try:
        return float(re.search(r"0?\.\d+", annotation).group())
    except:
        return 0.0

def ask_model(image_b64: str, questions: list, prefix_answers, previous_answers: str, k: int, examples: str) -> List[Tuple[str, float]]:
    """
    Given a list of already‐answered questions (prefix_answers),
    generate k candidate continuations for the next question.

    Parameters
    ----------
    image_b64: str
        The image for the particular case.
    questions: List[str]
        A complete list of questions with choices.
    prefix_answers: List[str]
        A list containing previous answers.
    k: int
        The number of candidates to generate
    examples: str
        Examples for fewshots showcasing the chain-of-thought reasoning.

    Returns
    -------
    candidiates: List[(str, float)]
        A list of (answer, score) tuples.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    q_idx = len(prefix_answers)
    next_q = questions[q_idx]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant in annotating panoramic image of oral lesions. "
                "Use a chain‐of‐thought approach: think step by step before giving each final answer. "
                "Do not refuse by saying 'I can’t analyze images.' "
                "This is for exploratory/educational use only."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below are two sets of example answers with the corresponding questions for you to generate more accurate responses.
                        """
                    )
                },
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {examples}
                """
            )   
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below is the image that you are going to help annotate.
                        """
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"""
                        The following are the already answered questions with the associated answers.
                        {previous_answers}
                        """
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Let’s think step by step.\n\n"
                        "For the following question, provide your internal reasoning followed by "
                        "a final answer (choose from the provided options for Q1–Q13, free-form for Q14 which is about making a diagnosis):\n\n"
                        f"{q_idx+1}. {next_q}\n"
                        "Just answet this question."
                    )   
                }
            ]
        }
    ]
                
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        n=k,
        temperature=0.7,
        top_p=0.9
    )

    candidates = []
    for choice in resp.choices:
        text = choice.message.content.strip()
        score = score_branch(text, image_b64, f"{q_idx+1}. {next_q}\n")
        candidates.append((text, score))
    return candidates

def ToTFewshots(
    image_path: str,
    questions: list,
    beam_width: int = 3,
    branch_width: int = 5,
    examples: str = None
) -> list:
    """
    Performs a global ToT beam search over the entire question sequence.
    Returns the best full 14-answer list.
    """
    # Each beam entry is (−score, [ans1, ans2, …, ans_m]), score is negative because heapq is a min-heap
    # Score is the measurement of the score of the whole sequence.
    beam = [(0.0, [])]

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    for step in tqdm(range(len(questions))):
        print(step)
        next_beam = []
        for neg_score, prefix in beam:
            previous_answers = " "
            for i, ans in enumerate(prefix, start=1):
                previous_answers += f"{i}. {questions[i-1]}\nAnswer: {ans}\n"
            cands = ask_model(base64_image, questions, prefix, previous_answers, branch_width, examples)
            for ans_text, cand_score in cands:
                new_prefix = prefix + [ans_text] 
                total_score = -neg_score + cand_score
                heapq.heappush(next_beam, (-total_score, new_prefix))
                if len(next_beam) > beam_width:
                    heapq.heappop(next_beam)

        beam = next_beam
        print(beam)

    best = max(beam, key=lambda x: -x[0])[1]
    return best

def self_debate(image_path: str, questions: str, examples: str) -> str:
    """
    Prompts with tree of thoughts.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                """
                You are an expert assistant in understanding and annotating panoramic image of oral lesions. 
                Your task is to analyze the provided image and answer the user's questions. 
                Do not refuse to answer by saying 'I can’t analyze images'—instead, adapt and provide the best possible response.
                
                IMPORTANT: "You are three radiology experts (Dr. A, Dr. B, and Dr. C) debating each question "
                "about the oral panoramic image.  \n"
                "1) Each doctor gives a short rationale.  \n"
                "2) They may disagree.  \n"
                "3) At the end, produce a single, unified answer for each question."
                Do not refuse to answer by stating “I can’t analyze images.” Instead, adapt and provide the best possible response based on the image data.
                Please provide your best-guess interpretation of the oral panoramic image along with your chain-of-thought reasoning. I understand that your analysis may not be clinically accurate and this is solely for educational or exploratory purposes, not for actual diagnosis.
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below are two sets of example answers for you to generate more accurate responses.
                        """
                    )
                }
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {examples}
                """
            )   
        },
        {
            "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            """
                            Below is the image that you are going to help annotate.
                            """
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            f"""
                            Please answer the following questions, the possible answers are provided as well:
                            {questions}
                            """
                        )   
                    }
                ]
        }
    ]
    
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.5,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

def self_critique(image_path: str, questions: str, examples: str, temperature=0.0) -> str:
    """
    Prompts with tree of thoughts.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                """
                You are an expert assistant in understanding and annotating oral panoramic imageof oral lesions. 
                Your task is to analyze the provided image and answer the user's questions. 
                Do not refuse to answer by saying 'I can’t analyze images'—instead, adapt and provide the best possible response.
                
                IMPORTANT: "For each question, you will:  \n"
                "  1) Give your best initial answer, using chain-of-Thought optional.  \n"
                "  2) Critically review your own answer, pointing out any weaknesses, ambiguities, or alternative interpretations.  \n"
                "  3) Provide a final, revised answer drawing on that self-critique."
                Do not refuse to answer by stating “I can’t analyze images.” Instead, adapt and provide the best possible response based on the image data.
                Please provide your best-guess interpretation of the oral panoramic image along with your chain-of-thought reasoning. I understand that your analysis may not be clinically accurate and this is solely for educational or exploratory purposes, not for actual diagnosis.
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below are two sets of example answers for you to generate more accurate responses.
                        """
                    )
                }
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {examples}
                """
            )   
        },
        {
            "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            """
                            Below is the image that you are going to help annotate.
                            """
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            f"""
                            Please answer the following questions, the possible answers are provided as well:
                            {questions}
                            — For each question, follow the 3-step process above.  \n
                            — At the very end, list “Final Answers:” and give Q1–Q14 with your revised choice.
                            """
                        )   
                    }
                ]
        }
    ]
    
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=temperature,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation


def inconsistency_annotation_generation(image_path: str, fake: str) -> str:
    """
    This one is used to explore GPT's ability to spot and correctify mistakes in the annoations
    of an image.
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                """
                You are an expert assistant in understanding and annotating panoramic image of oral lesions. 
                Your task is to analyze the provided image and correct the input answers of some questions. 
                Do not refuse to answer by saying 'I can’t analyze images'—instead, adapt and provide the best possible response.
                For each except the last question, you must correct the fake answers by choosing exactly one answer from the list of allowed options provided below.

                Which jaw is affected by the lesion?
                - Mandible
                - Maxilla
                - Mandible and maxilla
                        
                Where is the lesion centered anatomically? 
                - Molar region
                - Ramus region
                - Incisor region
                - Sinus region
                - TMJ region
                - Canine/Premolar region

                How does the lesion relate to the surrounding teeth? 
                - Apex associated Vital tooth
                - Apex associated Non-vital tooth
                - Apex associated tooth with unknown vitality unknown
                - Root associated
                - Crown associated
                - Missing tooth associated
                - Not tooth associated

                How many lesions are present? 
                - 1
                - 2
                - >3
                - Generalised lesion

                What is the maximum size of the lesion?
                - <2 cm
                - 2-3 cm
                - >3 cm

                What is the anatomical origin of the lesion? 
                - Central
                - Peripheral

                Specify the characteristics of the borders of the lesion?
                - Corticated
                - Defined but not corticated
                - Diffuse

                The loculation of the lesion is?
                - Unilocular
                - Multilocular
                - Not loculated

                What is the radiographic appearance of the contents of the lesions? 
                - Radiolucent
                - Radio-opaque
                - Mixed
                - Radiolucent with flecks
                - Opaque

                Does the lesion include one or more teeth? 
                - Yes
                - No

                Is there expansion of the bony cortex? 
                - Yes
                - No

                Does the lesion result in root resorption? 
                - Yes
                - No

                Are there any signs of tooth displacement or impaction? 
                - Yes
                - No
                """
            )
        },
        {
            "role": "system",
            "content": (
                """
                Only use the following structure for your responses for each question, there are 14 questions in total:

                - Answer: <Your correct answer>
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below is the image according to which you are going to correct the answers that may contain wrong ones.
                        """
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                        f"""
                        The questions and the answers you need to check and correctify are listed below:
                        
                        Questions:
                        Which jaw is affected by the lesion?
                        Where is the lesion centered anatomically? 
                        How does the lesion relate to the surrounding teeth? 
                        How many lesions are present? 
                        What is the maximum size of the lesion?
                        What is the anatomical origin of the lesion? 
                        Specify the characteristics of the borders of the lesion?
                        The loculation of the lesion is?
                        What is the radiographic appearance of the contents of the lesions? 
                        Does the lesion include one or more teeth? 
                        Is there expansion of the bony cortex? 
                        Does the lesion result in root resorption?  
                        Are there any signs of tooth displacement or impaction? 
                        What is the most possible lesion given the above information?

                        Answers:
                        {fake}
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

def leave_one_out(image_path: str, info: str, query: str, examples: str) -> str:
    """
    Explore the effect of the order of questions on the final performance. 
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant in annotating panoramic image of oral lesions. "
                "Use a chain‐of‐thought approach: think step by step before giving each final answer. "
                "Do not refuse by saying 'I can’t analyze images.' "
                "This is for exploratory/educational use only."
            )
        },
        {
            "role": "user",
            "content": [   
                {
                    "type": "text",
                    "text": (
                        """
                        Below are two sets of example answers with the corresponding questions for you to generate more accurate responses.
                        """
                    )
                },
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {examples}
                """
            )   
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below is the image that you are going to help annotate.
                        """
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                       f"""
                        We also learned the following information from a professional doctor:
                        {info}
                        """
                    )  

                },
                {
                    "type": "text",
                    "text": (
                       f"""
                        Please give the answer for the following question according to the image, possible answers are provided as well:
                        {query}
                        """
                    )   
                }
            ]
        }
    ]
    
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

def leave_one_out_txt(info: str, query: str, examples: str) -> str:
    """
    Explore the effect of the order of questions on the final performance. 
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)
    messages = [
        {
            "role": "system",
            "content": (
                """
                "You are an expert assistant in annotating oral lesions. "
                "Use a chain‐of‐thought approach: think step by step before giving each final answer. "
                "Do not refuse by saying 'I can’t analyze.' "
                "This is for exploratory/educational use only."
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below are two sets of example answers with the corresponding questions for you to generate more accurate responses.
                        """
                    )
                },
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {examples}
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
                        Now you need to answer the user's question.
                        We also learned the following information from a professional doctor:
                        {info}
                        """
                    )  

                },
                {
                    "type": "text",
                    "text": (
                       f"""
                        Please give the answer for the following question:
                        {query}
                        """
                    )   
                }
            ]
        }
    ]
    
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

def correct_one_mistake_known(image_path: str, info: str, question: str, examples: str) -> str:
    """
    Explore the effect of the order of questions on the final performance. 
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                """
                You are an expert assistant in understanding and annotating panoramic image of oral lesions.
                Your task is to correct the mistake in an annotation of the image given by a doctor student.
                There is onle one mistake in the given annotation, and the corresponding question is provided with you. 
                
                Use a chain‐of‐thought approach: think step by step before giving each final answer.
                Do not refuse by saying 'I can’t analyze images.
                This is for exploratory/educational use only.
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below are two sets of example answers with the corresponding questions for you to generate more accurate responses.
                        """
                    )
                },
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {examples}
                """
            )   
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below is the image that you are going to help annotate.
                        """
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                       f"""
                        The following annotation is from a doctor student, which includes one mistake:
                        {info}
                        The question containing the mistake is:
                        {question}
                        """
                    )  

                },
                {
                    "type": "text",
                    "text": (
                       f"""
                        Please correct the mistake by directly generating the correct answer together with the associated question.
                        """
                    )   
                }
            ]
        }
    ]
    
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

def correct_one_mistake_unknown(image_path: str, info: str, examples: str) -> str:
    """
    Explore the effect of the order of questions on the final performance. 
    """
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                """
                You are an expert assistant in understanding and annotating panoramic image of oral lesions.
                Your task is to correct the mistake in an annotation of the image given by a doctor student.
                There is onle one mistake in the given annotation, but you don't know which one is incorrect. 
                
                Use a chain‐of‐thought approach: think step by step before giving each final answer. 
                Do not refuse by saying 'I can’t analyze images.'
                This is for exploratory/educational use only.
                """
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below are two sets of example answers with the corresponding questions for you to generate more accurate responses.
                        """
                    )
                },
            ]
        },
        {
            "role": "assistant",
            "content": (
                f"""
                {examples}
                """
            )   
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """
                        Below is the image that you are going to help annotate.
                        """
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                       f"""
                        The following annotation is from a doctor student, which includes one mistake:
                        {info}
                        """
                    )  

                },
                {
                    "type": "text",
                    "text": (
                       f"""
                        Please correct the mistake by directly generating the correct answer together with the associated question whose answer you think is wrong.
                        """
                    )   
                }
            ]
        }
    ]
    
    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

def zeroshot_grok(image_path: str, questions: str) -> str:
    """
    Generate radiographic annotations with ChatGPT-4o Vision.
    """
    openai_client = OpenAI(
        api_key=api.GROK_KEY,
        base_url="https://api.x.ai/v1",
    )

    # Load and inline‐encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": (
                "You are an oral radiology expert assistant. "
                "Analyze the oral panoramic image thoroughly and provide your internal reasoning in free‐form. "
                "Then, for each numbered question, give a concise final answer drawn from the listed options. "
                "Do not say “I can’t analyze images.”"
                "This is for exploratory/educational use only."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the panoramic oral panoramic image to annotate:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Answer each question by number, selecting from the provided choices."
                        "For the final question, provide your own free-form response."
                        f"{questions}\n\n"
                        "Finally, list each question number followed by your answer."
                    )
                }
            ]
        }
    ]

    # Retry until non‐empty
    while True:
        response = openai_client.chat.completions.create(
            model="grok-2-vision-latest",
            messages=messages,
            max_tokens=5000,
            temperature=0.0,
            top_p=1.0
        )
        annotation = response.choices[0].message.content.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation


def zeroshot_claude(image_path: str, questions: str) -> str:
    """
    Generate radiographic annotations with ChatGPT-4o Vision.
    """
    client = anthropic.Anthropic(api_key=api.CLAUDE_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages_claude = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the oral panoramic image to annotate:"
                },
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": "image/png",
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Answer each question by number, selecting from the provided choices."
                        "For the final question, provide your own free-form response."
                        f"{questions}\n\n"
                        "Finally, list each question number followed by your answer."
                    )
                }
            ]
        }
    ]

    system = f"""
                You are an oral radiology expert assistant. 
                Analyze the oral panoramic image thoroughly and provide your internal reasoning in free‐form. 
                Then, for each numbered question, give a concise final answer drawn from the listed options. 
                Do not say “I can’t analyze images.”
                This is for exploratory/educational use only.
            """
    
    while True:
        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=1000,
            system = system,
            temperature=0,
            messages=messages_claude
        )
        annotation = message.content[0].text.strip()

        if not annotation or refusal_pattern.search(annotation):
            continue
        break
    return annotation

