import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
# os.environ["WANDB_MODE"] = "offline"
import time
import logging
import wandb
import argparse
import traceback
import torch
import torch.nn as nn
from collections import deque
from dataset import Dataset, custom_collate
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import get_cosine_schedule_with_warmup, Adafactor, BitsAndBytesConfig
from peft import BitFitConfig, TaskType, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict
from accelerate import infer_auto_device_map


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    logger.info(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}%") 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--dataset_name", type=str, default="ACL20")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--size_valid_set", type=int, default=10000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    
    parser.add_argument("--train_filename", type=str, default="/src")
    parser.add_argument("--dev_filename", type=str, default="/tgt")
    parser.add_argument("--output_dir", type=str, default="/save_model")
    
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--eval_step", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    parser.add_argument("--input_column_name", type=str, default="prompt")
    parser.add_argument("--output_column_name", type=str, default="completion")
    
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    # parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    # WandB 
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use WandB for logging")
    parser.add_argument("--wandb_project", type=str, default="JITCU_PEFT", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--logging_steps", type=int, default=10)

    return parser.parse_args()


def validation_step(epoch, model, validation_loader, save_dir, best_loss, device_ids, early_stop_counter, parallel=False,global_step=0):
    # print('-------start validation--------')
    logger.info("-------start validation--------")
    all_data = len(validation_loader)
    validation_loss = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            data = {
                'input_ids': data['input_ids'].to(device_ids[0]),
                'labels': data['labels'].to(device_ids[0]),
                'attention_mask': data['attention_mask'].to(device_ids[0])
            }
            output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
            loss = output.loss
            # if i % args.eval_step == 0:
            #     logger.info(" %d/%d , Val_Step_Loss = %s", i, all_data, loss.mean().item())
            validation_loss.append(loss.mean().item())
    
    test_loss = sum(validation_loss) / len(validation_loss)
    # print('validation loss:', round(sum(validation_loss) / len(validation_loss), 4))
    logger.info("Val_Avg_Loss = %s", test_loss)
    if args.use_wandb:
        wandb.log({
            "val/loss": test_loss,
            "val/best_loss": min(best_loss, test_loss)
        }, step=global_step)
    
    # if not parallel:
    #     model.module.save_pretrained(save_dir+'/Epoch_'+str(epoch+1))
    # else:
    #     model.save_pretrained(save_dir+'/Epoch_'+str(epoch+1))
    
    if test_loss < best_loss:
        best_loss = test_loss
        early_stop_counter = 0
        logger.info("Current is Best Loss = %s", test_loss)
        if not parallel:
            model.module.save_pretrained(save_dir+'/Best_Loss')
        else:
            model.save_pretrained(save_dir+'/Best_Loss')
    else:
        early_stop_counter += 1
        logger.info("Current Not Best Loss, Current Loss = %s, Best Loss is = %s", test_loss, best_loss)
    
    if epoch == epochs-1:
        # print('This is Last Checkpoint, Currect Loss=' + str(test_loss))
        logger.info("This is Last Checkpoint, Current Loss = %s, Best Loss is = %s", test_loss, best_loss)
        # if not parallel:
        #     model.module.save_pretrained(save_dir+'/Last')
        # else:
        #     model.save_pretrained(save_dir+'/Last')
    
    model.train()
    logger.info("---------end validation--------")
    
    return best_loss, early_stop_counter


def fine_tune(training_file, validation_file, epochs, batch_size, save_dir, device_ids, parallel=True, load_range=None):
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name if args.wandb_run_name else f"run-{time.strftime('%m%d-%H%M')}",
            config=vars(args)
        )
    logger.info("---------------------------")
    print('Load Tokenizer and Model...')
    
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("Loading model from:", os.path.abspath(args.model_name_or_path))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoderbase', trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    
    # model = prepare_model_for_int8_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = BitFitConfig(
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)
        
    logger.info("---------------------------")
    logger.info("Load Dataset...")
    
    training_dataset = Dataset(training_file, tokenizer, max_length=args.max_seq_length, shuffle=False, load_range=load_range)
    validation_dataset = Dataset(validation_file, tokenizer, max_length=args.max_seq_length, load_range=None)
    
    training_sampler = torch.utils.data.SequentialSampler(training_dataset)
    validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)
    
    training_loader = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=training_sampler, collate_fn=custom_collate
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=validation_sampler, collate_fn=custom_collate
    )

    # optimizer = torch.optim.SGD(model.parameters(), lr=2.5e-4, momentum=0.9)
    optimizer = Adafactor(model.parameters(),
                        lr=args.learning_rate,
                        scale_parameter=False,   
                        relative_step=False,      
                        warmup_init=False,        
                        weight_decay=0.005,        
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100, 
        num_training_steps=int(epochs * len(training_loader) / args.gradient_accumulation_steps)
    )
    
    # print('Model train...')
    logger.info("Model train...")


    #Start training
    logger.info("***** Running training *****")
    logger.info("      Batch size = %d", args.train_batch_size)
    logger.info("      Num epoch = %d", args.num_train_epochs)
    logger.info("****************************")

    best_loss = 100000000000
    early_stop_counter = 0
    stop_training = False

    for epoch in range(epochs):
        if stop_training: break
        
        logger.info("  epoch = %d", epoch)
        model.train()
        training_loss_window = deque(maxlen=100)
        start_time = time.time()
        oom = 0
        all_data = len(training_loader)

        for i, data in enumerate(training_loader):
            global_step = epoch * len(training_loader) + i
            data = {
                'input_ids': data['input_ids'].to(device_ids[0]),
                'labels': data['labels'].to(device_ids[0]),
                'attention_mask': data['attention_mask'].to(device_ids[0])
            }
            try:
                optimizer.zero_grad()
                output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
                loss = output.loss
                loss.mean().backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                current_step_loss = loss.mean().item()
                training_loss_window.append(current_step_loss)
                avg_loss = sum(training_loss_window) / len(training_loss_window)    

                if args.use_wandb and global_step % args.logging_steps == 0:
                    wandb.log({
                        "train/step_loss": loss.mean().item(),
                        "train/avg_loss":round(avg_loss, 4),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    }, step=global_step)
                    logger.info("Epoch: %d, %d/%d, Train_Step: %d, train/step_loss = %s, train/avg_loss = %s", epoch, i, all_data, global_step, loss.mean().item(), round(avg_loss, 4))
                
                if global_step > 0 and global_step % args.eval_step == 0:
                    best_loss, early_stop_counter = validation_step(                
                    epoch, model, validation_loader, save_dir, 
                    best_loss, device_ids, early_stop_counter, parallel, 
                    global_step=global_step 
                    )


                    if early_stop_counter >= args.patience:
                        logger.info("!!! Early stopping triggered after %d evaluations without improvement. Stop training.", args.patience)
                        stop_training = True
                        break 
                if stop_training: break 
                model.train() 

            except Exception as e:
                # print(str(e))
                logger.info(str(e))
                if 'out of memory' in str(e):
                    oom += 1
                model.zero_grad()
                optimizer.zero_grad()
                scheduler.step()
                del data

                torch.cuda.empty_cache()
        
        # if epoch % args.eval_step == 0:
        #     best_loss=validation_step(model, validation_loader, save_dir, best_loss=-1000000, parallel=parallel)


if __name__ == '__main__':
    args = get_args()
    logger.info(args)
    
    # Setup CUDA, GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    # args.device = device
    # logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    os.makedirs(args.output_dir, exist_ok=True)

    device_ids = [0]

    training_file = args.train_filename
    validation_file = args.dev_filename
    vocabulary_file = args.model_name_or_path
    pretrained_file = args.model_name_or_path
    
    epochs = args.num_train_epochs
    batch_size = args.train_batch_size
    save_dir = args.output_dir

    fine_tune(training_file, validation_file, epochs, batch_size, save_dir, device_ids=device_ids, parallel=True, load_range=None)