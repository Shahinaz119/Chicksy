import os

from dotenv import load_dotenv
from openai import OpenAI

from retrieve_context import retrieve_with_hybrid

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# ==========================
# Build Context
# ==========================

def build_context(query, top_k=5):

    retrieved = retrieve_with_hybrid(
        query,
        top_k=top_k
    )

    context = "\n\n".join(
        retrieved["document"].tolist()
    )

    return context, retrieved


# ==========================
# Prompt Template
# ==========================

def build_prompt(query, context):

    return f"""
You are an expert poultry farm assistant.

You are Chicksy, an AI poultry assistant.

Use ONLY the provided context to answer.

Provide a clear, concise, and professional answer.

Do not quote the retrieved text verbatim.

Summarize the relevant information naturally.

If the answer is not found in the context, say:
"I couldn't find this information in the handbook."

If the answer is not found,
reply with:

"I couldn't find this information in the handbook."

Context
-------
{context}

Question
--------
{query}

Answer:
"""


# ==========================
# Generate Answer
# ==========================

def generate_answer(
    query,
    model= "meta-llama/llama-3.2-1b-instruct"
):

    context, retrieved = build_context(query)
    print("=" * 50)
    print(context)
    print("=" * 50)
    prompt = build_prompt(
        query,
        context
    )

    response = client.chat.completions.create(

        model=model,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )

    return {

        "query": query,

        "answer":
        response.choices[0].message.content,

        "retrieved_chunks": retrieved

    }


# ==========================
# Test
# ==========================

if __name__ == "__main__":

    result = generate_answer(

        "What is crop fill?"

    )

    print("=" * 60)

    print(result["answer"])


    from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("OPENROUTER_API_KEY"))
