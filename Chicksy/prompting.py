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
You are Chicksy, an AI assistant specialized in broiler management.

Use ONLY the information in the provided context.

Instructions:
- Answer the question clearly and professionally.
- Never repeat the answer.
- Do NOT add extra information that is not in the context.
- If the context contains the answer, answer it directly.
- Answer in short paragraphs.
- Use bullet points when appropriate.
- Keep the answer under 120 words.
- Do not repeat information.
- ONLY if the context has absolutely no relevant information, reply exactly:
"I couldn't find this information in the handbook."

Context:
{context}

Question:
{query}

Answer:
"""


# ==========================
# Generate Answer
# ==========================

def generate_answer(
    query,
    model= "qwen/qwen3-8b"
):

    context, retrieved = build_context(query)
    print("=" * 50)
    print(context)
    print("=" * 50)
    print(context)
    prompt = build_prompt(
        query,
        context
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
           {
              "role": "user",
              "content": prompt
        }
    ]
)

    
    print(response.choices[0].message.content)
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
