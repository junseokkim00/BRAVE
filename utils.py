import re
from LLMs import set_model
from prompts import (
    QUESTIONDECOMPOSE_MSG,
    RAG_MSG,
    EVIDENCEEXTRACT_MSG,
    EVIDENCEEXTRACT_MULTI_MSG
)
from langchain.prompts import ChatPromptTemplate
from typing import List,Dict

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
            return num #TODO check if key is int or str
        except:
            print(f"attempt {attempt} failed.")
            attempt+=1
    return []



def run_RAG(model, context:str, question:str) -> str:
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


def postprocess(pred, label):
    if pred.endswith('.'):
        pred = pred[:-1]
    return pred.lower(), label.lower()

