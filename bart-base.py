import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset('hotpotqa/hotpot_qa', 'fullwiki', trust_remote_code=True)
train_dataset = ds['train']

def preprocess(example):
    context="context: "
    question = example['question']
    for title, passage in zip(example['context']['title'], example['context']['sentences']):
        if title in example['supporting_facts']['title']:
            sentences = " ".join(passage)
            context+=sentences
            context+='\n\n'
    context+=f'question: {question} answer: '
    example = {
        'contexts': context,
        'label': example['answer']
    }
    return example

train_dataset_mapped = train_dataset.map(preprocess)
train_dataset_mapped = train_dataset_mapped.select_columns(['contexts', 'label'])

train_dataset_mapped = train_dataset_mapped.train_test_split(test_size=0.2)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokenizer.model_max_length = 1024
def tokenize(batch):
    batch_encoding = tokenizer(
        batch['contexts'],
        padding="longest",
        truncation = True,
        return_tensors='pt'
    )
    target_encoding = tokenizer(
        batch['label'],
        padding="longest",
        truncation = True,
        return_tensors='pt'
    )
    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        'input_ids': batch_encoding['input_ids'],
        'attention_mask': batch_encoding['attention_mask'],
        'labels': labels
    }
def test_tokenize(batch):
    batch_encoding = tokenizer(
        batch['contexts'],
        padding="longest",
        truncation = True,
        return_tensors='pt'
    )
    target_encoding = tokenizer(
        batch['label'],
        padding="longest",
        truncation = True,
        return_tensors='pt'
    )
    labels = target_encoding.input_ids
    return {
        'input_ids': batch_encoding['input_ids'],
        'attention_mask': batch_encoding['attention_mask'],
        'labels': labels
    }

encoded_train_dataset = train_dataset_mapped['train'].map(tokenize, batch_size=8, batched=True)
encoded_test_dataset = train_dataset_mapped['test'].map(test_tokenize, batch_size=8, batched=True)
encoded_train_dataset = encoded_train_dataset.remove_columns(['contexts', 'label'])
encoded_test_dataset = encoded_test_dataset.remove_columns(['contexts', 'label'])

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors='pt')

train_dataset = DataLoader(encoded_train_dataset, batch_size=8, collate_fn=collate_fn)
val_dataset = DataLoader(encoded_test_dataset, batch_size=8, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
model.to('cuda')
max_em=0
for i in range(20):
    train_loss=0
    for batch in tqdm(train_dataset):
        batch.to('cuda')
        output = model(**batch)
        loss = output.loss
        train_loss += output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = train_loss / len(train_dataset)
    cnt=0
    total_cnt=0
    for batch in tqdm(val_dataset):
        batch.to('cuda')
        output = model.generate(**batch)
        for o, l in zip(output, batch['labels']):
            o = tokenizer.decode(o, skip_special_tokens=True)
            l = tokenizer.decode(l, skip_special_tokens=True)
            if o==l:
                cnt+=1
            total_cnt+=1
    print(f"last batch generated_output: {tokenizer.batch_decode(output, skip_special_tokens=True)}")
    print(f"last batch labels: {tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)}")
    print(f"epoch {i}: exact_match: {cnt / total_cnt * 100}")

    if max_em < cnt/ total_cnt * 100:
        max_em = cnt / total_cnt * 100
        model.save_pretrained('./bart-base-hotpotqa')

