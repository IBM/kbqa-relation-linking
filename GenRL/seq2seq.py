import re
import argparse
import ujson as json
from tqdm import tqdm
from collections import Counter
from transformers import BartTokenizer, BartForConditionalGeneration

def generate(tokenizer, model, paragraph, max_length=1024, num_beams=4, num_return_sequences=1, device='cpu'):
    inputs = tokenizer([paragraph], max_length=max_length, return_tensors='pt')
    inputs.to(device)
    keys_ids = model.generate(inputs['input_ids'], num_beams=num_beams, max_length=max_length, early_stopping=False, num_return_sequences=num_return_sequences)
    slots = [tokenizer.decode(k, skip_special_tokens=True, clean_up_tokenization_spaces=False) for k in keys_ids]
    return slots

def clean_record(record):
    record = record.replace('(', '[')
    record = record.replace(')', ']')
    record = record.replace(' vs ', ' | ')
    record = record.replace(' & ', ' | ')
    return record
    
def parse(record):
    record = clean_record(record)
    pairs = []
    r1 = re.findall(r"\[([^|]+)\|([^\]]+)\]", record)
    for pair in r1:
        pairs.append([pair[0].strip(), pair[1].strip()])
    return pairs
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file')
    parser.add_argument('--model_name')
    parser.add_argument('--device')
    parser.add_argument('--output')

    args = parser.parse_args()

    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    model.to(args.device)

    fout = open(args.output, 'w')
    for line in tqdm(open(args.test_file)):

        instance = json.loads(line)
        target_pred = generate(tokenizer, model, instance['source'], max_length=512, num_beams=50, num_return_sequences=50, device=args.device)
        relations = [parse(pred) for pred in target_pred]
        result = {'q_id': instance['q_id'] ,'question': instance['source'], 'relations': relations}
        fout.write(json.dumps(result) + '\n')
    fout.close()



if __name__ == "__main__":
    main()


