import Vision_Model_Exploration.api as api
from openai import OpenAI

def symptom_response(query, context):
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    messages_symptom = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who answers the user's question based only on the relevant portions "
                "of the provided context. Do not analyze or mention information that is not directly related "
                "to the user's question. Provide only the most likely disease name with its stage number, and a detailed explanation according to the given context."
                "Do not go beyond the provided context."
            )
        },
        {
            "role": "user",
            "content": (
                f"The user's question is: {query}\n\n"
                "Below is some background information, only partial information is related. Only use it if it helps directly answer the question:\n\n"
                f"{context}\n\n"
            )
        }
    ]

    response_symptom = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages_symptom,
        max_tokens=300,
    )

    symptom_explanation = response_symptom.choices[0].message.content
    return symptom_explanation

def symptom_list_response(query, context):
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    messages_symptom = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who answers the user's question based only on the relevant portions "
                "of the provided context. Do not analyze or mention information that is not directly related to the user's question. "
                "Provide only the most likely disease name and list all possible stage numbers or types, and a detailed explanation according to the given context."
                "Do not go beyond the provided context and notice to distinguish disease name from symptom descriptions.."
            )
        },
        {
            "role": "system",
            "content": (
                "Exactly use the following structure:"
                "Disease name:"
                "Stage numbers or types:"
                "Explanation:"
            )
        },
        {
            "role": "user",
            "content": (
                f"The user's question is: {query}\n\n"
                "Below is some background information, only partial information is related. Only use it if it helps directly answer the question:\n\n"
                f"{context}\n\n"
            )
        }
    ]

    response_symptom = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages_symptom,
        max_tokens=300,
    )

    symptom_explanation = response_symptom.choices[0].message.content
    return symptom_explanation