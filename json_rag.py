from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

print("Imports done")

# Step 1: Initialize LLM (Mistral via Ollama)
llm = ChatOllama(model="mistral")
print("LLM loaded")

# Step 2: Define the prompt template for tax advice
TAX_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an expert Indian tax advisor. A user has provided the following expense details:

    - Category: {category}
    - Amount: ₹{amount}
    - Relation: {relation}
    - Age: {age}
    - Description: {description}

    Based on the latest Indian Income Tax Act (FY 2024-25), provide:
    1. The applicable deduction section (like 80C, 80D, 80E, etc.)
    2. The max limit for this section
    3. How much the user can claim
    4. A simple explanation for the user

    Be clear and precise. If this is not deductible, say so clearly.
    """
)
print("Prompt defined")

# Step 3: Create the chain
chain = (
    {"category": RunnablePassthrough(),
     "amount": RunnablePassthrough(),
     "relation": RunnablePassthrough(),
     "age": RunnablePassthrough(),
     "description": RunnablePassthrough()}
    | TAX_PROMPT
    | llm
    | StrOutputParser()
)
print("Chain setup complete")

# Step 4: Sample user input (replace this with dynamic input from UI)
user_input = {
    "category": "Medical",
    "amount": 25000,
    "relation": "Parent",
    "age": 68,
    "description": "Health insurance premium for senior citizen father"
}

# Step 5: Run the LLM chain
response = chain.invoke(user_input)

# Step 6: Show the output
print("\n Tax Saving Suggestion:")
print(response)
