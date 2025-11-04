from torch.utils.data import DataLoader
from dataset.dataset import get_dataset
from transfomer_train import *
import torch
import numpy as np
import transformer
import naivebayes
from utils import shared_args

from tqdm import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import tensor_to_text

def compression_ratio_bounds(bpc, sequence_length, original_bits_per_symbol=8.0, overhead_bits=2.0):
    """
    Convert bits-per-token cross entropy into expected compression ratio bounds.
    Returns lower/upper bounds on (compressed bits)/(original bits).
    """
    if sequence_length <= 0 or original_bits_per_symbol <= 0:
        raise ValueError("sequence_length and original_bits_per_symbol must be positive.")
    per_token_upper = bpc + overhead_bits / float(sequence_length)
    lower_bound = bpc / float(original_bits_per_symbol)
    upper_bound = per_token_upper / float(original_bits_per_symbol)
    return lower_bound, upper_bound

class assemble_transformer(nn.Module):
    def __init__(self, args, alpha = 0.0):
        super(assemble_transformer, self).__init__()
        if args.load_model is None:
            config = transformer.TransformerConfig(embedding_dim=args.embedding_dim, num_heads=args.num_heads, num_layers=args.num_layers)
            self.model = transformer.TransformerDecoder(config)
        else:
            checkpoint = torch.load(
                args.load_model,
                map_location="cpu",
                weights_only=False,  # legacy checkpoints require full pickle load
            )
            config = checkpoint['config']
            self.model = transformer.TransformerDecoder(config)
            self.model.load_state_dict(checkpoint['model'])
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.naivebayes = naivebayes.NaiveBayes(base_weight=args.base_weight)
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))
        
    def update_alpha(self, alpha):
        with torch.no_grad():
            self.alpha.copy_(alpha)
        
    def forward(self, batch, get_all_probs=False):
        '''
        when alpha = 0, it is the transformer model
        when alpha = 1, it is the naivebayes model
        x: the original whole sequence, shape: (batch_size, seq_len), seq_len should be 2048
        '''
        
        bos_token = torch.zeros(batch.shape[0], 1, dtype=torch.long).to(batch.device)
        x = torch.cat([bos_token, batch], dim=1).to(batch.device)
        
        if self.alpha == 1.0:
            transformer_logits = 0.0
        else:
            transformer_logits = self.model(x[:,:-1])
        
        if self.alpha == 0.0:
            naivebayes_logits = 0.0
        else:
            naivebayes_logits = self.naivebayes(x[:,:-1])
            
        logits = (1 - self.alpha) * transformer_logits + self.alpha * naivebayes_logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        if get_all_probs:
            return log_probs
        else:
            target_log_probs = log_probs.gather(2, batch.unsqueeze(-1)).squeeze(-1)
            return target_log_probs

class assemble_llm(nn.Module):
    def __init__(self, args, alpha = 0.0):
        super(assemble_llm, self).__init__()
        self.args = args
        if args.model_type == 'gpt2':
            self.model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
            self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2") 
            self.vocab_size = self.tokenizer.vocab_size
        elif args.model_type == 'llama3-1B':
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            self.vocab_size = self.tokenizer.vocab_size + 2
        elif args.model_type == 'llama3-3B':
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
            self.vocab_size = self.tokenizer.vocab_size + 2
        elif args.model_type == 'llama3-8B':
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            self.vocab_size = self.tokenizer.vocab_size + 2
        else:
            raise ValueError(f"Model type {args.model_type} not supported.")
        
        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        for param in self.model.parameters():
            param.requires_grad = False

        self.naivebayes = naivebayes.NaiveBayes(self.vocab_size, base_weight=args.base_weight)
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))
    
    def update_alpha(self, alpha):
        with torch.no_grad():
            self.alpha.copy_(alpha)
            
    def get_text(self, batch):
        text_list = tensor_to_text(batch)
        if self.args.model_type == 'gpt2':
            new_text_list = []
            for text in text_list:
                new_text = self.tokenizer.bos_token + text
                new_text_list.append(new_text)
            return new_text_list
        elif self.args.model_type in ['llama3-1B', 'llama3-3B', 'llama3-8B']:
            return text_list
        else:
            raise ValueError(f"Model type {self.args.model_type} not supported.")

    def forward(self, batch, get_all_probs=False):
        '''
        when alpha = 0, it is a llm
        when alpha = 1, it is the naivebayes model
        x: the original whole sequence, shape: (batch_size, seq_len)
        '''
        inputs = self.tokenizer(self.get_text(batch), padding=True, truncation=True, return_tensors="pt").to(batch.device)
        with torch.no_grad():
            # add a condition to avoid the model to run twice when alpha = 1.0 or 0.0
            if self.alpha == 1.0:
                logits = 0.0
            else:
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits[:,:-1,:self.vocab_size]
                
            if self.alpha == 0.0:
                naivebayes_log_probs = 0.0
            else:
                naivebayes_log_probs = self.naivebayes(inputs["input_ids"])[:,1:,:]
            
        logits = (1 - self.alpha) * logits + self.alpha * naivebayes_log_probs
        log_probs = logits.log_softmax(dim=-1)
        log_probs = log_probs * inputs["attention_mask"][:,1:].unsqueeze(-1)
        if get_all_probs:
            return log_probs
        else:
            selected_log_probs = torch.gather(log_probs, -1, inputs["input_ids"][:,1:].unsqueeze(-1))
            return selected_log_probs
            
def alpha_train(args, model, batch, optimizer, scheduler):
    '''
    batch should be the original whole sequence, 
    shape: (batch_size, seq_len), seq_len should be 2048
    dtype: torch.long
    '''
    model.train()
    
    log_probs = model(batch)
    loss = - log_probs.sum()/batch.shape[-1]/batch.shape[0]/np.log(2)
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
    optimizer.step()
    scheduler.step()
    
    wandb.log({'grad_norm': cal_grad_norm(model),
                'batch-bpc': loss.item(), 
                'lr': optimizer.param_groups[0]['lr'],
                'alpha': model.alpha.item()})
        
def alpha_validate(args, alpha, model, loader):
    model.update_alpha(alpha)
    model.eval()
    mean_bpc = mean_logger()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='validating'):
            batch = batch.to(next(model.parameters()).device)
            log_probs = model(batch)
            bpc = -log_probs.sum().item()/batch.shape[-1]/batch.shape[0]/np.log(2)
            mean_bpc.update(bpc)
            
    return mean_bpc.get()
        
def min_(a,b):
    return a if a < b else b

if __name__ == '__main__':
    shared_args = shared_args()
    shared_args.add_argument('--rewrite_project_name', type=str, default='alpha_optimize')
    args = shared_args.get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model_type == 'transformer':
        model = assemble_transformer(args).to(device)
    elif args.model_type == 'gpt2':
        model = assemble_llm(args).to(device)
    elif args.model_type == 'llama3-1B':
        model = assemble_llm(args).to(device)
    elif args.model_type == 'llama3-3B':
        model = assemble_llm(args).to(device)
    elif args.model_type == 'llama3-8B':
        model = assemble_llm(args).to(device)
    else:
        raise ValueError(f"Model type {args.model_type} not supported.")
    
    if args.model_type == 'transformer':
        model_name = f'{args.model_type}_{args.embedding_dim}'
    else:
        model_name = f'{args.model_type}'
    
    run_name = f'alpha_optimize_naivebayes_{args.dataset}_{model_name}'
    
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.rewrite_project_name,
        # group=Group Name
        name=run_name,
        #document the args
        config=args
    )

    training_dataset, validation_dataset = get_dataset(
        dataset_name=args.dataset, 
        databudget_ratio=args.databudget_ratio, 
        databudget=args.databudget, 
        data_dir=args.data_dir
        )
    
    training_loader = cycle(DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4))
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    first_validation_sample = validation_loader.dataset[0]
    if isinstance(first_validation_sample, (list, tuple)):
        first_validation_sample = first_validation_sample[0]
    sequence_length = first_validation_sample.shape[-1] if isinstance(first_validation_sample, torch.Tensor) else len(first_validation_sample)
    
    def log_compression_bounds(prefix, bpc_value):
        lower, upper = compression_ratio_bounds(bpc_value, sequence_length)
        return {
            f'{prefix}_compression_ratio_lower': lower,
            f'{prefix}_compression_ratio_upper': upper,
        }
    
    naivebayes_bpc = alpha_validate(args, 1., model, validation_loader)    
    pretrained_bpc = alpha_validate(args, 0., model, validation_loader)
    
    wandb.log({
        'pretrained_bpc': pretrained_bpc,
        'naivebayes_bpc': naivebayes_bpc,
        **log_compression_bounds('pretrained', pretrained_bpc),
        **log_compression_bounds('naivebayes', naivebayes_bpc),
    })
    
    model.update_alpha(0.0)
    optimizer = optim.Adam([model.alpha], lr=args.learning_rate)
    # Warm-up scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_iteration)
    # Cosine annealing scheduler (starts after the warm-up)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iteration - args.warmup_iteration, eta_min=args.learning_rate / 4)
    # Combine the warm-up and cosine annealing using torch.optim.lr_scheduler.SequentialLR
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_iteration])
    
    for i in tqdm(range(args.iteration + 1), desc='Training iteration'):
        batch = next(training_loader).to(device)
        alpha_train(args, model, batch, optimizer, scheduler)
    
    best_alpha = model.alpha.item()
    best_bpc = alpha_validate(args, best_alpha, model, validation_loader)
        
    wandb.log({'best_alpha': best_alpha,
                'best_bpc': best_bpc,
                'improvement': min_(pretrained_bpc, naivebayes_bpc) - best_bpc,
                'improvement_rate': (min_(pretrained_bpc, naivebayes_bpc) - best_bpc)/min_(pretrained_bpc, naivebayes_bpc),
                **log_compression_bounds('best', best_bpc),
                })
