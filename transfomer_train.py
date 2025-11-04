import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import get_dataset
from utils import *
import transformer
import naivebayes
from itertools import cycle
import os
from tqdm import tqdm
import wandb

# Function to compute the loss
def compute_loss(logits, targets):
    # logits: (batch_size, sequence_length, vocab_size)
    # targets: (batch_size, sequence_length)
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    # Mean over the sequence length
    mean_log_probs = target_log_probs.mean(dim=1)  # Shape (batch_size,)
    loss = -mean_log_probs.mean()
    return loss

def transformer_train(args, model, batch, optimizer, scheduler, device):
    model.train()
    
    bos_token = torch.zeros(batch.shape[0], 1, dtype=torch.long)
    batch = torch.cat([bos_token, batch], dim=1)
    batch = batch.to(device)
    logits = model(batch[:,:-1])
    loss = compute_loss(logits, batch[:,1:])
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
    optimizer.step()
    scheduler.step()
    
    if args.en_wandb:
        wandb.log({'loss': loss.item(), 
                    'grad_norm': cal_grad_norm(model),
                    'bpd': loss.item() / np.log(2), 
                    'lr': optimizer.param_groups[0]['lr']})

def transformer_test(args, model, validation_data_loader, device):
    model.eval()
    mean_bpd = mean_logger()
    
    for batch in tqdm(validation_data_loader, desc='validating'):
        bos_token = torch.zeros(batch.shape[0], 1, dtype=torch.long)
        batch = torch.cat([bos_token, batch], dim=1)
        batch = batch.to(device)
        logits = model(batch[:,:-1])
        loss = compute_loss(logits, batch[:,1:])
        
        # Compute bits per dimension (bpd)
        bpd = loss.item() / np.log(2)
        mean_bpd.update(bpd)
                
        if args.evaluate_only and args.en_wandb:
            wandb.log({"average_bpd": mean_bpd.get()})
        
    return mean_bpd.get()

if __name__ == '__main__':
    shared_args = shared_args()
    args = shared_args.get_args()
    
    if args.model_type == 'transformer':
        get_model = transformer.get_model
        train = transformer_train; test = transformer_test
        run_name = f'train_{args.dataset}_dim{args.embedding_dim}_lr{args.learning_rate}_{args.shortnotes}'
        project_name = 'language_modelling_is_compression'
    elif args.model_type == 'naivebayes':
        get_model = naivebayes.get_model
        train = transformer_train; test = transformer_test
        if args.context_window_length is not None:
            run_name = f'dataset_{args.dataset}_context_window_{args.context_window_length}_base_weight_{args.base_weight}_{args.shortnotes}'
        else:
            run_name = f'dataset_{args.dataset}_context_window_Full_base_weight_{args.base_weight}_{args.shortnotes}'
        project_name = 'naivebayes'
    else:
        raise ValueError(f"Model type {args.model_type} not supported.")

    if args.load_model is not None and not args.evaluate_only:
        run_name = f'{run_name}_finetuned'
        
    if args.en_wandb:
        if args.evaluate_only:
            run_tag = "evaluate_only"
        else:
            run_tag = "training"
    
        wandb.init(
            # set entity to specify your username or team name
            # entity="qihangz-work",
            # set the wandb project where this run will be logged
            project=f'{project_name}_{run_tag}',
            # group=Group Name
            name=run_name,
            # set tags for this run
            tags=args.tags,
            # set notes for this run
            notes=args.notes,
            #document the args
            config=args
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = get_model(args, args.load_model, device)
    
    if args.evaluate_only:
        en_shuffle = False
    else:
        en_shuffle = True
    
    training_dataset, validation_dataset = get_dataset(dataset_name=args.dataset)
    training_loader = cycle(DataLoader(training_dataset, batch_size=args.batch_size, shuffle=en_shuffle))
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    try:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        # Warm-up scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_iteration)
        # Cosine annealing scheduler (starts after the warm-up)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iteration - args.warmup_iteration, eta_min=args.learning_rate / 4)
        # Combine the warm-up and cosine annealing using torch.optim.lr_scheduler.SequentialLR
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_iteration])
    except Exception as e:
        print(e)
        optimizer = None
        warmup_scheduler = None
        cosine_scheduler = None
        scheduler = None
        
    best_bpd = float('inf')
    list_of_eval = list_of_evaluations(args.iteration)
    for i in tqdm(range(args.iteration + 1), desc='Training iteration'):
        batch = next(training_loader)
        if not args.evaluate_only:
            train(args, model, batch, optimizer, scheduler, device)
        
        if i in list_of_eval:
            with torch.no_grad():
                validation_bpd = test(args, model, validation_loader, device)
                if args.en_wandb:
                    wandb.log({'validation_bpd': validation_bpd,
                               'training_budget': round(i / args.iteration, 3),
                            })
                    
                if validation_bpd < best_bpd:
                    best_bpd = validation_bpd
                    if not args.evaluate_only:
                        if args.en_save:
                            checkpoint = {
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'dataset': args.dataset,
                            'iteration': i,
                            'validation_bpd': validation_bpd,
                            'config': config
                            }
                        
                            if not os.path.exists(args.save_dir):
                                os.makedirs(args.save_dir)
                            
                            torch.save(checkpoint, os.path.join(args.save_dir, f'{run_name}.pth'))
                            print("=" * 40)
                            print('Model parameters saved.')
                            print("Validation bpd: ", validation_bpd)
                            print("=" * 40)
            if args.evaluate_only:
                break
        
    if args.en_wandb:        
        wandb.log({'best_bpd': best_bpd})