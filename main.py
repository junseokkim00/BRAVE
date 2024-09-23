import argparse
from tqdm import tqdm
from datasets import load_dataset
from utils import questionDecompose, evidenceExtractor, postprocess, multiEvidenceExtractor, singleEvidenceExtraction, no_context_llm, select_RAG
from LLMs import set_model
import random
import json

def brave_me(model, dataset, args):
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
            for i in num:
                relevance_num.add(str(i))
        # generate find_grained_context
        relevance_num = list(relevance_num)
        for num in relevance_num:
            if num in context_dict:
                fine_grained_context+=context_dict[num]
                fine_grained_context+='\n\n'

        pred = select_RAG(args.generation_model_name)(model=model, context=fine_grained_context, question=question, args=args)
        pred, answer = postprocess(pred, label=answer)
        preds.append(pred)
        labels.append(answer)
        with open(f'./{args.run_type}_{args.model_name}_{args.dataset}_{args.split}_k:{args.num_data}_seed:{args.seed}{args.metadata}.jsonl', 'a') as f:
            json_file = {
                'id': total_cnt,
                'question': question,
                'answer': answer,
                'pred': pred,
                'subQuestions': subQuestions,
                'original_context': context,
                'fine_grained_context': fine_grained_context,
                'chosen_titles': [data['context']['title'][int(eval(num))-1] for num in relevance_num],
                'gold_chosen_titles': data['supporting_facts']['title'],
                'correct': pred == answer
            }
            f.write(json.dumps(json_file)+'\n')
        print(f"pred: {pred}\tlabel: {answer}")
        if pred == answer:
            cnt+=1
        total_cnt+=1
    return cnt, total_cnt

def brave_wo_QD(model, dataset, args):
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
        for i in num:
            relevance_num.add(str(i))
        # generate find_grained_context
        relevance_num = list(relevance_num)
        for num in relevance_num:
            if num in context_dict:
                fine_grained_context+=context_dict[num]
                fine_grained_context+='\n\n'

        pred = select_RAG(args.generation_model_name)(model=model, context=fine_grained_context, question=question, args=args)
        pred, answer = postprocess(pred, label=answer)
        preds.append(pred)
        labels.append(answer)
        with open(f'./{args.run_type}_{args.model_name}_{args.dataset}_{args.split}_k:{args.num_data}_seed:{args.seed}{args.metadata}.jsonl', 'a') as f:
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


def brave(model, dataset, args):
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

        pred = select_RAG(args.generation_model_name)(model=model, context=fine_grained_context, question=question, args=args)
        pred, answer = postprocess(pred, label=answer)
        preds.append(pred)
        labels.append(answer)
        with open(f'./{args.run_type}_{args.model_name}_{args.dataset}_{args.split}_k:{args.num_data}_seed:{args.seed}{args.metadata}.jsonl', 'a') as f:
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


def no_context(model, dataset, args):
    cnt=0
    total_cnt=0
    preds=[]
    labels=[]
    for data in tqdm(dataset):
        question, answer = data['question'], data['answer']
        pred = no_context_llm(model=model, question=question)
        pred, answer = postprocess(pred, label=answer)

        preds.append(pred)
        labels.append(answer)
        with open(f'./{args.run_type}_{args.model_name}_{args.dataset}_{args.split}_k:{args.num_data}_seed:{args.seed}{args.metadata}.jsonl', 'a') as f:
            json_file = {
                'id': total_cnt,
                'question': question,
                'answer': answer,
                'pred': pred,
                'correct': pred == answer
            }
            f.write(json.dumps(json_file)+'\n')
        print(f"pred: {pred}\tlabel: {answer}")
        if pred == answer:
            cnt+=1
        total_cnt+=1
    return cnt, total_cnt

def rag(model, dataset, args):
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
        pred = select_RAG(args.generation_model_name)(model=model, context=context, question=question, args=args)
        pred, answer = postprocess(pred, label=answer)
        
        preds_og.append(pred)
        labels_og.append(answer)
        with open(f'./{args.run_type}_{args.model_name}_{args.dataset}_{args.split}_k:{args.num_data}_seed:{args.seed}{args.metadata}.jsonl', 'a') as f:
            json_file = {
                'id': total_cnt,
                'question': question,
                'answer': answer,
                'pred': pred,
                'fine_grained_context': context,
                'correct': pred == answer
            }
            f.write(json.dumps(json_file)+'\n')
        
        print(f"pred: {pred}\tlabel: {answer}")
        if pred == answer:
            cnt+=1
        total_cnt+=1
    return cnt, total_cnt

def brave_bert(model, dataset, args):
    cnt=0
    total_cnt=0
    preds_bert=[]
    labels_bert=[]
    for data in tqdm(dataset):
        question, answer, sentences = data['question'], data['answer'], data['context']['sentences']
        contexts = []
        relevance_titles=[]
        for sent, title in zip(sentences, data['context']['title']):
            sent = ' '.join(sent)
            decision = singleEvidenceExtraction(context=sent, question=question)
            if decision:
                contexts.append(sent)
                relevance_titles.append(title)



        print(f"{total_cnt+1}/{len(dataset)} Question: {question}")
        context=""
        for idx, sent in enumerate(contexts):
            context+=sent
            context+='\n\n'
        pred = select_RAG(args.generation_model_name)(model=model, context=context, question=question, args=args)
        pred, answer = postprocess(pred, label=answer)
        
        preds_bert.append(pred)
        labels_bert.append(answer)
        with open(f'./{args.run_type}_{args.model_name}_{args.dataset}_{args.split}_k:{args.num_data}_seed:{args.seed}{args.metadata}.jsonl', 'a') as f:
            json_file = {
                'id': total_cnt,
                'question': question,
                'answer': answer,
                'pred': pred,
                'fine_grained_context': context,
                'chosen_titles': relevance_titles,
                'gold_chosen_titles': data['supporting_facts']['title'],
                'correct': pred == answer
            }
            f.write(json.dumps(json_file)+'\n')
        
        print(f"pred: {pred}\tlabel: {answer}")
        if pred == answer:
            cnt+=1
        total_cnt+=1
    return cnt, total_cnt


def oracle(model, dataset, args):
    cnt=0
    total_cnt=0
    preds=[]
    labels=[]
    for data in tqdm(dataset):
        question, answer, relevant_title = data['question'], data['answer'], data['supporting_facts']['title']
        contexts=[]
        for title, sentences in zip(data['context']['title'], data['context']['sentences']):
            if title in relevant_title:
                sent = ' '.join(sentences)
                contexts.append(sent)
        
        print(f"{total_cnt+1}/{len(dataset)} Question: {question}")
        context=""
        for sent in contexts:
            context+=sent
            context+='\n\n'
        pred = select_RAG(args.generation_model_name)(model=model, context=context, question=question, args=args)
        pred, answer = postprocess(pred, label=answer)

        preds.append(pred)
        labels.append(answer)
        with open(f'./{args.run_type}_{args.model_name}_{args.dataset}_{args.split}_k:{args.num_data}_seed:{args.seed}{args.metadata}.jsonl', 'a') as f:
            json_file = {
                'id': total_cnt,
                'question': question,
                'answer': answer,
                'pred': pred,
                'gold_chosen_titles': data['supporting_facts']['title'],
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
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_data', type=int)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--generation_model_name', type=str, choices=['llm', 'bart'], default='llm')
    parser.add_argument('--run_type', type=str, choices=['no_context','brave', 'rag', 'plain', 'brave_wo_QD', 'brave_me', 'brave_bert', 'oracle'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--metadata', type=str, default='')
    args = parser.parse_args()

    # variables
    dataset_name=args.dataset
    dataset_split = args.dataset_split
    k = args.num_data
    name = args.model_name
    seed = args.seed
    run_type = args.run_type
    split = args.split

    # set dataset
    if dataset_name == '2wikimh':
        with open(f'./2wikimh_revised/{split}.json', 'r') as f:
            ds = json.load(f)
        indices = random.sample(range(len(ds)), k)
        test_ds = [ds[idx] for idx in indices]
        args.dataset = '2wikimh'
        if args.generation_model_name == 'bart':
            args.bart_path = 'bart-base-2wiki'
    
    else:
        ds = load_dataset(dataset_name, dataset_split, trust_remote_code=True)
        indices = random.sample(range(len(ds['train'])), k)
        test_ds = ds['train'].select(indices)
        args.dataset = 'hotpotQA'
        if args.generation_model_name == 'bart':
            args.bart_path = 'bart-base-hotpotqa'

    model = set_model(name)


    run_type_dict = {
        'brave': brave,
        'brave_wo_QD': brave_wo_QD,
        'brave_me': brave_me,
        'rag': rag,
        'brave_bert': brave_bert,
        'no_context': no_context,
        'oracle': oracle
    }
    cnt, total_cnt = run_type_dict[run_type](model=model, dataset=test_ds, args=args)

    print(f"exact_match: {cnt}/ {total_cnt} = {cnt / total_cnt}")



