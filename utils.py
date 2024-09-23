import re
from LLMs import set_model
from prompts import (
    QUESTIONDECOMPOSE_MSG,
    RAG_MSG,
    EVIDENCEEXTRACT_MSG,
    EVIDENCEEXTRACT_MULTI_MSG,
    NO_CONTEXT_MSG
)
from langchain.prompts import ChatPromptTemplate
from typing import List,Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartForConditionalGeneration, BartTokenizer
import torch



def select_RAG(model_name):
    if model_name == 'llm':
        return run_RAG
    else:
        return bart_RAG


def bart_RAG(model, context, question, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    tokenizer.model_max_length = 1024
    tokenizer.truncation_side="left"
    bart_model = BartForConditionalGeneration.from_pretrained(f"./{args.bart_path}")
    batch = tokenizer(f"context: {context}question: {question} answer: ", truncation=True, return_tensors='pt')
    bart_model.to(device)
    batch.to(device)
    output = bart_model.generate(**batch)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    return output[0]

def singleEvidenceExtraction(context, question):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained('./models/roberta-base-single-extraction-100')
    batch = tokenizer(f"Context: {context}{tokenizer.unk_token}Question: {question}", truncation=True, return_tensors='pt')
    model.to(device)
    with torch.no_grad():
        batch.to(device)
        outputs = model(**batch)
    pred = outputs.logits.argmax(axis=-1).detach().cpu()
    pred = pred.item()
    return pred


# llm version
def questionDecompose(model, question:str, max_attempt=10) -> List:
    attempt=0
    while attempt < max_attempt:
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("user", QUESTIONDECOMPOSE_MSG)
            ])
            chain = chat_prompt | model

            output = chain.invoke({
                'question': question,
            }).content
            output = eval(output[output.index('['):output.index(']')+1])
            return output
        except:
            print(f"attempt {attempt} failed. Try again.")
            attempt+=1
    return []
    


def evidenceExtractor(model, context: str, context_ls: Dict, question: str):
    chat_prompt = ChatPromptTemplate.from_messages([
        ("user", EVIDENCEEXTRACT_MSG)
    ])
    chain = chat_prompt | model
    output = chain.invoke({
        'question': question,
        'context': context
    }).content
    msg = EVIDENCEEXTRACT_MSG + output + 'Now, please provide only the number that you chose without any explanation or preambles. '
    answer_extraction = ChatPromptTemplate.from_messages([
        ("user", msg)
    ])
    chain_ans = answer_extraction | model
    final_output = chain_ans.invoke({
        'question': question,
        'context': context
    }).content
    num = re.findall('\d+', final_output)
    return num[0] #TODO check if key is int or str

def multiEvidenceExtractor(model, context: str, context_ls: Dict, question: str, max_attempt=10):
    attempt = 0
    while attempt < max_attempt:
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("user", EVIDENCEEXTRACT_MULTI_MSG)
            ])
            chain = chat_prompt | model
            output = chain.invoke({
                'question': question,
                'context': context
            }).content
            # msg = EVIDENCEEXTRACT_MULTI_MSG + output + 'Now, please provide only the number that you chose without any explanation or preambles. '
            # answer_extraction = ChatPromptTemplate.from_messages([
            #     ("user", msg)
            # ])
            # chain_ans = answer_extraction | model
            # final_output = chain_ans.invoke({
            #     'question': question,
            #     'context': context
            # }).content
            num = output[output.index('['):output.index(']')+1]
            num = eval(num)
            return num #TODO check if key is int or str
        except:
            print(f"attempt {attempt} failed.")
            attempt+=1
    return []



def run_RAG(model, context:str, question:str, args) -> str:
    chat_prompt = ChatPromptTemplate.from_messages([
        ("user", RAG_MSG)
    ])
    chain = chat_prompt | model
    # first invoke
    output = chain.invoke({
        'question': question,
        'context': context
    })
    print(output.content)
    msg = RAG_MSG + output.content + 'Now, please provide only the answer without any explanation or preambles. Therefore, the answer is '
    answer_extraction = ChatPromptTemplate.from_messages([
        ("user", msg)
    ])
    chain_ans = answer_extraction | model
    # second invoke for returning only the output
    final_output = chain_ans.invoke({
        'question': question,
        'context': context
    }).content
    return final_output

def no_context_llm(model, question: str):
    chat_prompt = ChatPromptTemplate.from_messages([
        ("user", NO_CONTEXT_MSG)
    ])
    chain = chat_prompt | model
    # first invoke
    output = chain.invoke({
        'question': question,
    })
    print(output.content)
    msg = NO_CONTEXT_MSG + output.content + 'Now, please provide only the answer without any explanation or preambles. Therefore, the answer is '
    answer_extraction = ChatPromptTemplate.from_messages([
        ("user", msg)
    ])
    chain_ans = answer_extraction | model
    # second invoke for returning only the output
    final_output = chain_ans.invoke({
        'question': question
    }).content
    return final_output


def postprocess(pred, label):
    if pred.endswith('.'):
        pred = pred[:-1]
    return pred.lower(), label.lower()


def compute_precision_recall(targets, predictions, k):
    """
    targets : 테스트 데이터 중, 사용자의 피드백이 있는 item 인덱스 리스트
    predictions : 학습 데이터에 피드백이 존재하는 item을 제외한 예측 데이터
    k : 추천 수
    """
    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall

