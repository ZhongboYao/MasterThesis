import Vision_Model_Exploration.api as api
from openai import OpenAI

def disease_list_prediction(query, context):
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    messages_baseline = [
        {
            "role": "system",
            "content": (
                "You are a professional doctor who can predict the patient's lesion name given the description of the patient. Predictions shoud be based on the relevant portions "
                "of the provided context."
                "Provide only the 3 most likely lesion names and the corresponding description:, ignoring the stage number."
                "Do not go beyond the provided context and notice to distinguish disease name from symptom descriptions."
                "Exactly use the following structure:"
                "Disease name 1:"
                "Disease 1 description:"
                "Disease name 2:"
                "Disease 2 description:"
                "Disease name 3:"
                "Disease 3 description:"
            )
        },
        {
            "role": "user",
            "content": (
                f"The user's question is: {query}\n\n"
                "Below is some background information, only partial information is related. Only use it if it helps directly answer the question:\n\n"
                f"{context}\n\n"
                "Exactly use the following structure:"
                "Disease name 1:"
                "Disease 1 description:"
                "Disease name 2:"
                "Disease 2 description:"
                "Disease name 3:"
                "Disease 3 description:"
            )
        }
    ]

    response_symptom = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_baseline,
        max_tokens=300,
    )

    symptom_explanation = response_symptom.choices[0].message.content
    return symptom_explanation

def query_enrichment(query, context):
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    messages_symptom = [
        {
            "role": "system",
            "content": (
                "You are an expert medical assistant tasked with converting keyword-based patient symptom descriptions "
                "into clear, detailed, and medically accurate narratives. Your goal is to enhance clarity and readability, "
                "enabling healthcare professionals to quickly and effectively understand patient conditions. Utilize provided "
                "contextual information on diseases to support accurate and relevant rephrasing."
            )
        },
        {
            "role": "user",
            "content": (
                f"Original keyword-based description:\n{query}\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Relevant disease contexts for reference:\n{context}\n"
            )
        }
    ]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_symptom,
        max_tokens=300,
    )

    new_query = response.choices[0].message.content
    return new_query