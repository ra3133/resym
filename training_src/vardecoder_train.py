import torch
from dataset import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import argparse
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import os

hf_key = os.environ['HF_TOKEN']


def count_dataset_samples(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            count += 1
    return count


def train(train_fpath, save_dir, model_name, max_token, lr, epochs, batch_size, bf16, log_steps):
    kwargs = DistributedDataParallelKwargs(static_graph=True, find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    print(f'==========start loading model {model_name} ==========')
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_key)
    if bf16:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, use_auth_token=hf_key, torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, use_auth_token=hf_key
        )
        
    model.train()
    model.transformer.gradient_checkpointing = True
    model.config.use_cache = False

    max_token = min(max_token, tokenizer.model_max_length)

    train_dataset = Dataset(train_fpath, tokenizer, max_len=max_token, shuffle=True)

    dataset_size = count_dataset_samples(train_fpath)
    num_devices = torch.cuda.device_count()
    print(f"[INFO] Using {num_devices} GPU(s)")

    total_steps_per_epoch = dataset_size / (batch_size * num_devices)
    num_save_step = max(1, round(0.20 * total_steps_per_epoch))

    trainer_kwargs = dict(
        output_dir=save_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type='cosine',
        warmup_steps=500,
        gradient_accumulation_steps=1,
        num_train_epochs = epochs,
        gradient_checkpointing=True,
        optim='adamw_torch',
        save_strategy='steps',
        save_steps=num_save_step,
        logging_dir='./logs',
        logging_strategy='steps',
        logging_steps=log_steps,
        prediction_loss_only=True,
    )
    if bf16:
        trainer_kwargs['bf16'] = True
    else:
        trainer_kwargs['fp16'] = True

    trainer_args = TrainingArguments(**trainer_kwargs)
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_fpath')
    parser.add_argument('save_dir')
    parser.add_argument('--model_name', type=str, default='bigcode/starcoderbase-3b')
    parser.add_argument('--max_token', type=int, default=4096, help='Maximum total context length')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Per-device batch size')
    parser.add_argument('--bf16', action='store_true', help='Enable bfloat16 training mode')
    parser.add_argument('--log_steps', type=int, default=10, help='Number of steps for log')

    args = parser.parse_args()
    train(
        train_fpath=args.train_fpath,
        save_dir=args.save_dir,
        model_name=args.model_name,
        max_token=args.max_token,
        lr=args.lr,
        epochs=args.epoch,
        batch_size=args.batch_size,
        bf16=args.bf16,
        log_steps=args.log_steps
    )