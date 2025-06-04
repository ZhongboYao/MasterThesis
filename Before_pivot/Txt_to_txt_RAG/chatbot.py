import Vision_Model_Exploration.api as api
from openai import OpenAI

def disease_list_prediction(query, context):
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    messages_baseline = [
        {
            "role": "system",
            "content": (
                "You are a professional doctor who can predict the patient's lesion name given the description of the patient."
                "Provide only the 3 most likely oral lesion names and the corresponding description, ignoring the stage number."
                "Some contexts are provided that may support you for the diagnosis."
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
                f"The user's symptom description is: {query}\n\n"
                "Below is some background information, some of them may be or not helpful in predicting the oral lesion names:\n\n"
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