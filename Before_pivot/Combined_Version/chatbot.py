import Vision_Model_Exploration.api as api
from openai import OpenAI

def disease_list_prediction(query, context):
    openai_client = OpenAI(api_key=api.OPENAI_KEY)

    messages_symptom = [
        {
            "role": "system",
            "content": (
                "You are a professional doctor who can predict the patient's lesion name given the description of the patient. Predictions shoud be based only on the relevant portions "
                "of the provided context. Do not analyze or mention information that is not directly related to the user's question. "
                "Provide only the 3 most likely lesion names and the corresponding description:, ignoring the stage number, and don't give anything else."
                "Do not go beyond the provided context and notice to distinguish disease name from symptom descriptions.."
            )
        },
        {
            "role": "system",
            "content": (
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


# def query_enrichment(query, context):
#     openai_client = OpenAI(api_key=api.OPENAI_KEY)

#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a professional doctor assistant who can rephrase the patient's keywords description "
#                 "into an enriched description so that it can be better understood by doctors. "
#                 "Use the provided examples to guide your rephrasing."
#             )
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"The user's original query is: {query}\n\n"
#                 "Below are some examples for the rephrased queries:\n\n"
#                 f"{context}\n\n"
#             )
#         }
#     ]

#     response = openai_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         max_tokens=300,
#     )

#     final_response = response.choices[0].message.content
#     return final_response