# 0. Import the required packages
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_openai import AzureChatOpenAI

from langchain_prompty import create_chat_prompt
from pathlib import Path

# 1. Create a llm
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

# 2. Create a classify chain(Recognize user intent)
classify_chain = (
    PromptTemplate.from_template(
        """Classify the user's request into one of the following categories: 
        `pronunciation`, `grammar`, `translation`.
        <question>
        {question}
        </question>
        Category:"""
    )
    | llm
    | StrOutputParser()
)

# 3. Create multiple chains using Prompty
folder = Path(__file__).parent.absolute().as_posix()
chains = {}
prompty_files = ["translator.prompty", "grammar.prompty", "pronunciation.prompty"]
chain_names = ["translation_chain", "grammar_chain", "pronunciation_chain"]

for prompty_file, chain_name in zip(prompty_files, chain_names):
    path_to_prompty = folder + "/" + prompty_file
    prompt = create_chat_prompt(path_to_prompty)
    chains[chain_name] = prompt | llm

# 4. Create a branch
branch = RunnableBranch(
    (lambda x: "pronunciation" in x["topic"].lower(), chains["pronunciation_chain"]),
    (lambda x: "grammar" in x["topic"].lower(), chains["grammar_chain"]),
    chains["translation_chain"]
)

full_chain = ({"topic": classify_chain, "question": lambda x: x["question"]} 
              | branch 
              | StrOutputParser())

response = full_chain.invoke({"question": "'どうもありがとうございます', Could you translate this sentence?"})

print("====================================")
print(response)
print("====================================")
