"""Utility functions."""
import argparse

class mean_logger:
    def __init__(self):
        self.mean = 0
        self.n = 0
    def update(self, val):
        self.mean = (self.mean * self.n + val) / (self.n + 1)
        self.n += 1
    def get(self):
        return self.mean
    
class ratio_logger:
    def __init__(self):
        self.nume = 0
        self.deno = 0
        self.current_deno = 0
        self.current_nume = 0

    def update(self, current_deno, current_nume):
        self.current_deno = current_deno
        self.current_nume = current_nume
        self.deno += current_deno
        self.nume += current_nume

    def get_rate(self):
        if self.deno == 0:
            return 0
        return (self.nume / self.deno)

    def get_current_rate(self):
        if self.current_deno == 0:
            return 0
        return (self.current_nume / self.current_deno) 

    def get_total_file_size(self):
        return self.deno, self.nume
    
def list_of_evaluations(max_iter):
    list_of_eval = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    list_of_eval += [i /10 for i in range(11)]
    list_of_eval.sort()
    list_of_eval = [int(round(i * max_iter)) for i in list_of_eval]
    
    return list_of_eval

def cal_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
    return total_norm

def tensor_to_text(batch):
    """
    Convert a tensor of token encodings to text.
    
    Args:
        batch (torch.Tensor or array-like): A tensor of shape (batch_size, seq_len), where each element is a token encoding.                                               
    Returns:
        If batch size is 1, return a string;
        If batch size is greater than 1, return a list of strings, each corresponding to the text converted from a sequence.
    """
    # Convert each sequence: for each value in the sequence, if it is greater than 0, convert it to the corresponding character and concatenate it into a string
    text_list = [''.join(chr(int(token)) for token in sequence if token > 0) for sequence in batch]
    
    # Return the result based on the batch size
    return text_list

class shared_args:
    def __init__(self) -> None:
        self.argparser = argparse.ArgumentParser()
        self.initalize_parser()
    
    def initalize_parser(self):                
        # data I/O
        self.argparser.add_argument('--data_dir', 
                                    type=str, 
                                    default='./data', 
                                    help='Location for the dataset')
        
        self.argparser.add_argument('--save_dir', 
                                    type=str, 
                                    default='../model',
                                    help='Location for parameter checkpoints and samples')
        self.argparser.add_argument('--en_save', 
                                    type=bool, 
                                    default=False,
                                    help='Enable saving model ckpts')
        
        self.argparser.add_argument('--output_dir', 
                                    type=str, 
                                    default=None,
                                    help='Location for compressed code')
        
        self.argparser.add_argument('--dataset', 
                                    type=str, 
                                    default='enwik8', 
                                    help='Can be either enwik8|enwik9|code|math|shakespeare')
        self.argparser.add_argument('--load_model', type=str, default=None, 
                                    help='load trained autoregressive model')
        self.argparser.add_argument('--databudget', type=int, default=None, 
                                    help='The number of training data')
        self.argparser.add_argument('--databudget_ratio', type=float, default=None,
                                    help='The ratio of training data')
        
        # model_type 
        self.argparser.add_argument('--model_type', 
                                    type=str, 
                                    default='transformer',
                                    help='Model type: transformer|gpt2|llama3')
        
        # transformer
        self.argparser.add_argument('--embedding_dim', 
                                    type=int, 
                                    default=64, 
                                    help='Number of dimensions in embedding')
        self.argparser.add_argument('--num_layers', 
                                    type=int, 
                                    default=4,
                                    help='Number of Decoder layers in the model')
        self.argparser.add_argument('--num_heads', 
                                    type=int, 
                                    default=8,
                                    help='Number of heads in multihead attention')
        
        # Naive Bayes
        self.argparser.add_argument('--context_window_length', 
                                    type=int, 
                                    default=None,
                                    help='The context window length')
        self.argparser.add_argument('--context_kernel_size', type=int, default=11,
                                    help='The kernel size for the context window')
        self.argparser.add_argument('--base_weight', type=float, default=1.0,
                                    help='The weight for the base model')
        self.argparser.add_argument('--sigma', type=float, default=None,
                                    help='the hyperparameter to control how smooth the distribution is')
        
        #Training
        self.argparser.add_argument('--evaluate_only', type=bool, default=False,
                                    help='Whether to only evaluate the model')
        self.argparser.add_argument('--evaluate_interval', type=int, default=1000,
                                    help='evaluate interval')
        self.argparser.add_argument('--learning_rate', type=float, default=1e-4, 
                                    help='Base learning rate')
        self.argparser.add_argument('--batch_size', type=int, default=64,
                                    help='Batch size during training per GPU')
        self.argparser.add_argument('--iteration', type=int, default=60, 
                                    help='How many iterations to run in total?')
        self.argparser.add_argument('--warmup_iteration', type=int, default=10, 
                                    help='Number of iterations for linear warmup')
        self.argparser.add_argument('--max_norm', type=float, default=5.0, 
                                    help='Clipping gradient norm')
        self.argparser.add_argument('--seed', type=int, default=0,
                                    help='Random seed to use')
    
    def add_argument(self, *args, **kwargs):
        self.argparser.add_argument(*args, **kwargs)
    
    def get_args(self):
        return self.argparser.parse_args()