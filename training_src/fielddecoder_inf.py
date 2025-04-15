import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import time
from tqdm import tqdm
import os 

hf_key = os.environ['HF_TOKEN']

MAX_OUTPUT_TOKEN=1024
def inference(test_fpath, out_fpath, model_path, model_name, max_token, num_beams):
    print(f'==========start loading FieldDecoder {model_name} ==========')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_key)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, use_auth_token=hf_key,
        torch_dtype=torch.bfloat16, device_map='auto'
    )
    model.eval()
    wp = open(out_fpath, 'w')

    with open(test_fpath, 'r') as fp:
        for i, line in enumerate(tqdm(fp.readlines())):
            line = json.loads(line)
            try:
                prompt = line['prompt'] 
                first_token = line['first_token']
            except:
                continue

            start_time = time.time()
            with torch.no_grad():
                input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()[:, : max_token - MAX_OUTPUT_TOKEN]
                output = model.generate(
                    input_ids=input_ids, 
                    max_new_tokens=MAX_OUTPUT_TOKEN, 
                    num_beams=num_beams, 
                    num_return_sequences=1, 
                    do_sample=False,
                    early_stopping=False, 
                    pad_token_id=tokenizer.eos_token_id, 
                    eos_token_id=tokenizer.eos_token_id
                )[0]
                output = tokenizer.decode(output[input_ids.size(1): ], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                output = first_token + ':' + output

            time_used = time.time() - start_time
            save_data = line
            save_data['predict'] = output
            save_data['time'] = time_used
            wp.write(json.dumps(save_data) + '\n')

    print(f"Inference for FieldDecoder finished. The results can be found in {out_fpath}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_fpath')
    parser.add_argument('out_fpath')
    parser.add_argument('model_path')
    parser.add_argument('--model_name', type=str, default='bigcode/starcoderbase-3b')
    parser.add_argument('--max_token', type=int, default=8192, help='Maximum total context length (input + output)')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search generation')
    args = parser.parse_args()
    inference(args.test_fpath, args.out_fpath, args.model_path, args.model_name, args.max_token, args.num_beams)