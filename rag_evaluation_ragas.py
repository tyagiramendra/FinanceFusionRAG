## LLM as Evaluater 

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas import evaluate
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from dotenv import load_dotenv
load_dotenv()
from rag_pipeline import rag_with_sources
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils import clean_text
from ragas.metrics import faithfulness,answer_relevancy, answer_correctness, context_recall, context_precision
import pandas as pd
from datasets import Dataset


#Generate Sethetic Test Data
#load documents for evalution
loader = PyPDFDirectoryLoader("D:/projects/FinanceFusionRAG/data/eva_data",glob="*.pdf")
documents= loader.load()

for index,doc in enumerate(documents):
   documents[index].page_content = clean_text(documents[index].page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 450,chunk_overlap = 50)
documents = text_splitter.split_documents(documents)

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=20, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}, with_debugging_logs=True)
                                                 
test_df = testset.to_pandas()
test_questions = test_df["question"].values.tolist()
test_groundtruths = test_df["ground_truth"].values.tolist()
test_df.to_csv("test_data.csv",index=False)
print("Test Data has been generated!")

#Naive RAG Evalution #
df =pd.read_csv("test_data.csv")
test_questions = df["question"].values.tolist()
test_groundtruths = df["ground_truth"].values.tolist()

answers = []
contexts = []

for question in test_questions:
  response,docs=rag_with_sources(question)
  answers.append(response)
  contexts.append([doc["text"] for doc in docs])



response_dataset = Dataset.from_dict({
    "question" : test_questions,
    "answer" : answers,
    "contexts" : contexts,
    "ground_truth" : test_groundtruths
})

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
 
]

results = evaluate(response_dataset, metrics,raise_exceptions=False)
df_results = results.to_pandas()
df_results.to_csv("results.csv",index=False)

