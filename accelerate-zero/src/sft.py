"""Code for finetune_huatuo"""

import os
os.environ["WANDB_API_KEY"] = "3bf17abb57350d9a57be1dd3d2c4354780849cc6"  # xidong账号log
import copy
import json
import torch
import logging
import argparse
from typing import Literal, List, Dict, Sequence, Any, Tuple, Set

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
import transformers
from transformers import set_seed, get_cosine_schedule_with_warmup
import datasets
import shutil
import json
import random
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
# from models.tokenization_moss import MossTokenizer
os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

 
class RoleName:
    PATIENT = '<病人>'
    DOCTOR = '<HuatuoGPT>'

class SFT_Data(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, data):
        self.config = config
        self.tokenizer = tokenizer
        self.model_name = config.model_path
        self.data = data
        self.datacollatorforseq2seq = transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
        self.IGNORE_INDEX = -100
        self.system_prompt = f"{RoleName.PATIENT}和智能医疗大模型{RoleName.DOCTOR}之间的对话。对于用户的医疗问诊，{RoleName.DOCTOR}给出准确的、详细的、温暖的指导建议。对于用户的指令问题，{RoleName.DOCTOR}给出有益的、详细的、有礼貌的回答。"
        self.ROUND_END_SIGNAL = f' {self.tokenizer.eos_token}'
        self.UTTER_END_SIGNAL = " "
        self.debug = True

    def _encode_text(self, text, role, is_patient_utterance):
        text_with_role = f'{role}：{text}'
        end_signal = self.ROUND_END_SIGNAL if not is_patient_utterance else self.UTTER_END_SIGNAL
        return self.tokenizer(text_with_role + end_signal, add_special_tokens=False)['input_ids']

    def preprocess(self, data):
        # 初始化编码数据和标签
        encoded_data = self.tokenizer(self.system_prompt, add_special_tokens=False)['input_ids']
        label = [self.IGNORE_INDEX] * len(encoded_data)
        
        # 如果数据长度是奇数，则删除最后一个元素
        if len(data) % 2 == 1:
            data = data[:-1]
        
        # 角色名称
        role_names = [RoleName.PATIENT, RoleName.DOCTOR]
        
        # 遍历数据并编码
        for ind, d in enumerate(data):
            is_patient_utterance = ind % 2 == 0
            role = role_names[ind % 2]
            encode = self._encode_text(d, role, is_patient_utterance)
            encoded_data += encode
            if is_patient_utterance:
                label += [self.IGNORE_INDEX] * len(encode)
            else:
                label += encode
        
        # debug
        if self.debug:
            print(self.tokenizer.decode(encoded_data))
            self.debug = False
        
        # 返回处理后的数据
        return {'input_ids': encoded_data[:self.config.max_seq_len], 'labels': label[:self.config.max_seq_len]}


    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch):
        processed_batch = [self.preprocess(x) for x in batch]
        return self.datacollatorforseq2seq(processed_batch)

    def __len__(self):
        return len(self.data)



class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss
    

def train(args):
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_accumulation_steps) 
    if args.not_shuffle_train_loader:
        accelerator.print('Will not shuffle train data loader.')

    if accelerator.is_main_process:
        wandb.init(project = args.experiment_name, config=args, dir=args.log_dir)

    # if accelerator.distributed_type == 'DEEPSPEED':
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu*dist.get_world_size()*accelerator.gradient_accumulation_steps

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if 'chatglm-6b' in args.model_path or 'chatglm2-6b' in args.model_path:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.unk_token

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    with open(args.data_dir) as f:
        data = json.load(f)
    
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:1000]
    val_data = data[-48:]
    
    accelerator.print(f'train_data shuffle: {(not args.not_shuffle_train_loader)}')
    train_dataset = SFT_Data(args, tokenizer, train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.eval_bsz_per_gpu, shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)
    val_dataset = SFT_Data(args, tokenizer, val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_bsz_per_gpu, shuffle=False, drop_last=True, collate_fn=val_dataset.collate_fn)

    num_training_steps = (len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)

    accelerator.print(f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} data_dir:{args.data_dir} lr:{args.learning_rate} num_training_steps:{num_training_steps}')

    if args.checkpoint_path:
        if os.path.isfile(os.path.join(args.checkpoint_path, "scheduler.bin")) and \
           os.path.isfile(os.path.join(args.checkpoint_path, "training_state.pt")):
            accelerator.print(f"Loading trained model :{args.checkpoint_path}")
            accelerator.load_state(args.checkpoint_path)
            training_state = torch.load(os.path.join(args.checkpoint_path, "training_state.pt"))
            start_epoch = training_state["epoch"]
            start_step = training_state["step"]+1
            global_step = training_state["global_step"]
            accelerator.print(f"Checkpoint Loaded at {start_epoch} epoch, {start_step} step and {global_step} global step")

        else:
            raise ValueError(f"Checkpoint not found at: {args.checkpoint_path}")
    else:
        start_epoch = 0
        start_step = 0
        global_step = 0
    


    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    # Set args.save_step and args.eval_step to epoch step num and epoch step num//10 when two are -1
    if args.save_step==-1:
        args.save_step=len(train_dataloader)
        accelerator.print(f'Save step setted to {args.save_step}')
    if args.eval_step==-1:
        args.eval_step=len(train_dataloader)//10
        accelerator.print(f'Eval step setted to {args.eval_step}')

    # global_step = 0
    metric = SFTMetric(device=torch.cuda.current_device())

    #Code for saving checkpoints
    def save_checkpoint(epoch, step, global_step):
        #check ckpt nums and delete the oldest
        if accelerator.is_main_process:
            checkpoint_files = os.listdir(args.output_dir)
            checkpoint_files = [file for file in checkpoint_files if file.startswith("checkpoint-")]
            num_checkpoints = len(checkpoint_files)
            if args.max_ckpts>1:
                if num_checkpoints >= args.max_ckpts:
                    checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                    oldest_checkpoint = checkpoint_files[0]
                    shutil.rmtree(os.path.join(args.output_dir, oldest_checkpoint))
        accelerator.wait_for_everyone()

        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")

        try:
            if accelerator.state.deepspeed_plugin.zero_stage==3:
                unwrap_model = accelerator.unwrap_model(model)
                unwrap_model.save_pretrained(os.path.join(save_dir, f'tfmr'),is_main_process=accelerator.is_main_process,save_function=accelerator.save,state_dict=accelerator.get_state_dict(model))
            else:
                if accelerator.is_main_process:
                    model.save_pretrained(os.path.join(save_dir, f'tfmr'),state_dict=accelerator.get_state_dict(model))
                    accelerator.print('Save pretrained model with non-zero3 strategy')
            if accelerator.is_main_process:
                tokenizer.save_pretrained(os.path.join(save_dir, f'tfmr'))
        except Exception as e:
            accelerator.print(str(e))
            accelerator.print('Save pretrained model failed, skipped.')


        os.makedirs(save_dir, exist_ok=True)
        
        accelerator.save_state(save_dir)
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))
        accelerator.print(f'checkpoint checkpoint-{epoch}-{global_step} is saved...')

    accelerator.print(accelerator.deepspeed_config)
    model.train()
    for epoch in range(start_epoch, args.n_epochs):
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_cnt, batch in train_dataloader_iterator:
            if epoch==start_epoch and batch_cnt<start_step:
                # print(batch_cnt,start_step,epoch)
                continue
            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids=batch['input_ids']
            attention_mask=batch['attention_mask']
            labels=batch['labels']

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss = output.loss

            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()

            accelerator.backward(loss)
            
            if (global_step+1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                

            global_step += 1

            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=epoch, current_step=batch_cnt, total_step=len(train_dataloader), skip=accelerator.optimizer_step_was_skipped, loss=round(train_loss, 3), acc=round(acc, 3), length=len(input_ids[0]), lr=lr_scheduler.get_last_lr()[0])

            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': train_loss,
                    'acc': acc,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)

            if global_step % args.eval_step == 0:
                torch.cuda.empty_cache()
                model.eval() 

                val_metric = SFTMetric(torch.cuda.current_device())
                val_dataloader_iterator = tqdm(val_dataloader, total=len(val_dataloader)) if accelerator.is_main_process else val_dataloader
                for batch in val_dataloader_iterator:
                    input_ids=batch['input_ids']
                    attention_mask=batch['attention_mask']
                    labels=batch['labels']
                    with torch.no_grad():
                        output = model(return_dict=True,**batch)

                    val_metric(output.logits, labels, output.loss)

                val_acc, val_loss = val_metric.get_metric()

                if accelerator.is_main_process:
                    wandb.log({
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }, step=global_step)
                    accelerator.print(f"Epoch: {epoch}, Step: {batch_cnt}, Val loss: {val_loss}, Val acc: {val_acc}")

                model.train()           

            if global_step % args.save_step == 0:
                accelerator.wait_for_everyone()
                save_checkpoint(epoch, batch_cnt, global_step)
        start_step = 0
    wandb.finish()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Experiment Args
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--checkpoint_path',default=None, type=str)

    # Model Args
    parser.add_argument('--model_path', default='/mntnfs/med_data5/chenjunying/models/bloomz-7b1-mt', type=str)
    
    # Data Args
    parser.add_argument('--not_shuffle_train_loader', action='store_true')
    parser.add_argument('--data_dir', default='/mntcephfs/data/med/chenjunying/dataset/pre_training/pretrainv2/dataset', type=str)
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=5, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    
    # Training Args
    parser.add_argument('--max_seq_len', default=2048, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=1, type=int)
    parser.add_argument('--eval_bsz_per_gpu', default=4, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=2, type=int)

    # Other Args
    parser.add_argument('--save_step', default=3000, type=int)
    parser.add_argument('--eval_step', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir,args.experiment_name)
    args.output_dir = os.path.join(args.output_dir,args.experiment_name)


    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)           


