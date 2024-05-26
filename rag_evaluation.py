## Senthetic dataset has been generated by LLM , Please check rag_evaluation_pipeline.ipynb
## LLM as Evaluater 
import os
from config import set_environment, setup_langsmith
set_environment()
#setup_langsmith()

import constants as c

import langsmith
from langchain import chat_models, prompts, smith
from langchain.schema import output_parser
from rag_pipeline import rag_with_sources, combine_chunks
import pandas as pd
from tqdm.auto import tqdm
import json

# Evalution dataset is generated and validated by LLM. Please refer this notebook rag_evaluation_pipeline. 
eval_dataset = pd.read_csv("eval_dataset.csv")
rag_outputs = []
eval_dataset = eval_dataset.to_dict("records")
for example in tqdm(eval_dataset):
    question = example["question"]
    if question in [output["question"] for output in rag_outputs]:
            continue

    answer, relevant_docs = rag_with_sources(question)
    result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_doc"],
            "generated_answer": answer,
            "retrieved_docs": [doc["text"] for doc in relevant_docs],
        }
    rag_outputs.append(result)
    with open(c.output_file, "w") as f:
            json.dump(rag_outputs, f)