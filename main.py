import argparse
from tqdm import tqdm
from datasets import load_dataset
from utils import questionDecompose, evidenceExtractor, run_RAG, postprocess, multiEvidenceExtractor
from LLMs import set_model
import random
import json

def brave_me(model, dataset):
    total_cnt=0
    cnt=0
    preds=[]
    labels=[]
    for data in tqdm(dataset):
        question, answer, sentences = data['question'], data['answer'], data['context']['sentences']
        print(f"{total_cnt+1}/{len(test_ds)} Question: {question}")
        subQuestions=questionDecompose(model, question)
        print(f"subQuestions: {subQuestions}")
        fine_grained_context=""
        relevance_num=set()
        context=""
        context_dict={}
        for idx, sent_ls in enumerate(sentences):
            context+=f"{idx}. "
            context+=" ".join(sent_ls)
            context_dict[str(idx)] = " ".join(sent_ls)
            context+='\n\n'
        
        # extract evidence number
        for subQ in subQuestions:
            num = multiEvidenceExtractor(model, context, context_dict, subQ)
            num = eval(num)
            for i in num:
                relevance_num.add(str(i))
        # generate find_grained_context
        relevance_num = list(relevance_num)
        for num in relevance_num:
            fine_grained_context+=context_dict[num]
            fine_grained_context+='\n\n'

        pred = run_RAG(model=model, context=fine_grained_context, question=question)
        pred, answer = postprocess(pred, label=answer)
        preds.append(pred)
        labels.append(answer)
        with open(f'./brave_me_{name}_hotpotQA_train_k:{k}_seed:{seed}.jsonl', 'a') as f:
            json_file = {
                'id': total_cnt,
                'question': question,
                'answer': answer,
                'pred': pred,
                'subQuestions': subQuestions,
                'original_context': context,
                'fine_grained_context': fine_grained_context,
                'chosen_titles': [data['context']['title'][int(num)-1] for num in relevance_num],
                'gold_chosen_titles': data['supporting_facts']['title'],
                'correct': pred == answer
            }
            f.write(json.dumps(json_file)+'\n')
        print(f"pred: {pred}\tlabel: {answer}")
        if pred == answer:
            cnt+=1
        total_cnt+=1
    return cnt, total_cnt

def brave_wo_QD(model, dataset):
    total_cnt=0
    cnt=0
    preds=[]
    labels=[]
    for data in tqdm(dataset):
        question, answer, sentences = data['question'], data['answer'], data['context']['sentences']
        print(f"{total_cnt+1}/{len(test_ds)} Question: {question}")
        fine_grained_context=""
        relevance_num=set()
        context=""
        context_dict={}
        for idx, sent_ls in enumerate(sentences):
            context+=f"{idx}. "
            context+=" ".join(sent_ls)
            context_dict[str(idx)] = " ".join(sent_ls)
            context+='\n\n'
        
        # extract evidence number
        
        num = multiEvidenceExtractor(model, context, context_dict, question)
        num = eval(num)
        for i in num:
            relevance_num.add(str(i))
        # generate find_grained_context
        relevance_num = list(relevance_num)
        for num in relevance_num:
            fine_grained_context+=context_dict[num]
            fine_grained_context+='\n\n'

        pred = run_RAG(model=model, context=fine_grained_context, question=question)
        pred, answer = postprocess(pred, label=answer)
        preds.append(pred)
        labels.append(answer)
        with open(f'./brave_wo_QD_{name}_hotpotQA_train_k:{k}_seed:{seed}.jsonl', 'a') as f:
            json_file = {
                'id': total_cnt,
                'question': question,
                'answer': answer,
                'pred': pred,
                'subQuestions': None,
                'original_context': context,
                'fine_grained_context': fine_grained_context,
                'chosen_titles': [data['context']['title'][int(num)-1] for num in relevance_num],
                'gold_chosen_titles': data['supporting_facts']['title'],
                'correct': pred == answer
            }
            f.write(json.dumps(json_file)+'\n')
        print(f"pred: {pred}\tlabel: {answer}")
        if pred == answer:
            cnt+=1
        total_cnt+=1
    return cnt, total_cnt


def brave(model, dataset):
    total_cnt=0
    cnt=0

    preds=[]
    labels=[]
    for data in tqdm(dataset):
        question, answer, sentences = data['question'], data['answer'], data['context']['sentences']
        print(f"{total_cnt+1}/{len(test_ds)} Question: {question}")
        subQuestions=questionDecompose(model, question)
        print(f"subQuestions: {subQuestions}")
        fine_grained_context=""
        relevance_num=set()
        context=""
        context_dict={}
        for idx, sent_ls in enumerate(sentences):
            context+=f"{idx}. "
            context+=" ".join(sent_ls)
            context_dict[idx] = " ".join(sent_ls)
            context+='\n\n'
        
        # extract evidence number
        for subQ in subQuestions:
            num = evidenceExtractor(model, context, context_dict, subQ)
            relevance_num.add(num)

        # generate find_grained_context
        relevance_num = list(relevance_num)
        for num in relevance_num:
            fine_grained_context+=context_dict[num]
            fine_grained_context+='\n\n'

        pred = run_RAG(model=model, context=fine_grained_context, question=question)
        pred, answer = postprocess(pred, label=answer)
        preds.append(pred)
        labels.append(answer)
        with open(f'./brave_{name}_hotpotQA_train_k:{k}_seed:{seed}.jsonl', 'a') as f:
            json_file = {
                'id': total_cnt,
                'question': question,
                'answer': answer,
                'pred': pred,
                'subQuestions': subQuestions,
                'original_context': context,
                'fine_grained_context': fine_grained_context,
                'chosen_titles': [data['context']['title'][int(num)-1] for num in relevance_num],
                'gold_chosen_titles': data['supporting_facts']['title'],
                'correct': pred == answer
            }
            f.write(json.dumps(json_file)+'\n')
        print(f"pred: {pred}\tlabel: {answer}")
        if pred == answer:
            cnt+=1
        total_cnt+=1
    return cnt, total_cnt

def rag(model, dataset):
    cnt=0
    total_cnt=0
    preds_og=[]
    labels_og=[]
    for data in tqdm(dataset):
        question, answer, sentences = data['question'], data['answer'], data['context']['sentences']
        print(f"{total_cnt+1}/{len(test_ds)} Question: {question}")
        context=""
        for idx, sent_ls in enumerate(sentences):
            context+=" ".join(sent_ls)
            context+='\n\n'
        pred = run_RAG(model=model, context=context, question=question)
        pred, answer = postprocess(pred, label=answer)
        
        preds_og.append(pred)
        labels_og.append(answer)
        with open(f'./rag_{name}_hotpotQA_train_k:{k}_seed:{seed}.jsonl', 'a') as f:
            json_file = {
                'id': total_cnt,
                'question': question,
                'answer': answer,
                'pred': pred,
                'original_context': context,
                'correct': pred == answer
            }
            f.write(json.dumps(json_file)+'\n')
        
        print(f"pred: {pred}\tlabel: {answer}")
        if pred == answer:
            cnt+=1
        total_cnt+=1
    return cnt, total_cnt




if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hotpotqa/hotpot_qa')
    parser.add_argument('--dataset_split', type=str, default="fullwiki")
    parser.add_argument('--num_data', type=int)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--run_type', type=str, choices=['brave', 'rag', 'plain', 'brave_wo_QD', 'brave_me'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    # variables
    dataset_name=args.dataset
    dataset_split = args.dataset_split
    k = args.num_data
    name = args.model_name
    seed = args.seed
    run_type = args.run_type

    # set dataset
    ds = load_dataset(dataset_name, dataset_split, trust_remote_code=True)
    indices = random.sample(range(len(ds['train'])), k)
    test_ds = ds['train'].select(indices)

    model = set_model(name)


    run_type_dict = {
        'brave': brave,
        'brave_wo_QD': brave_wo_QD,
        'brave_me': brave_me,
        'rag': rag,
    }
    cnt, total_cnt = run_type_dict[run_type](model=model, dataset=test_ds)

    print(f"exact_match: {cnt}/ {total_cnt} = {cnt / total_cnt}")



