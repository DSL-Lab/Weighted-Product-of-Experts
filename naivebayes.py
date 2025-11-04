import torch
import torch.nn as nn

class NaiveBayes(nn.Module):
    def __init__(self, alphabet_size=256, context_window=None, base_weight=0.5):
        super(NaiveBayes, self).__init__()
        self.alphabet_size = alphabet_size
        self.base = torch.ones(alphabet_size).to(torch.float32).unsqueeze(0).unsqueeze(0) * base_weight
        
        if context_window == None:
            self.context_window = int(1e8)
        else:
            self.context_window = context_window
        
    def forward(self, x):
        '''
        x: (batch_size, sequence_length), 
        x[:,0] is the bos token
        the input is the sequence with the bos token
        '''
        # remove the bos token
        x = x[:,1:]
        one_hot_e = torch.nn.functional.one_hot(x, num_classes=self.alphabet_size).to(torch.float32)
        if self.context_window > x.shape[1]:
            counts = one_hot_e.cumsum(dim=1)
        else:
            n = x.shape[1]
            mask = torch.tril(torch.ones((n, n), dtype=torch.float32), diagonal=0) - torch.tril(torch.ones((n, n), dtype=torch.float32), diagonal=-self.context_window)
            one_hot_e = one_hot_e.unsqueeze(1)
            mask = mask.unsqueeze(0).unsqueeze(-1).to(one_hot_e.device)
            counts = (mask * one_hot_e).sum(dim=2)
        
        base = torch.zeros_like(counts[:,0,:]).to(torch.float32).view(-1,1,self.alphabet_size)
        counts = torch.cat([base, counts], dim=1) + self.base.to(counts.device)
        frequency = counts/(counts.sum(dim=-1, keepdim=True))
        return torch.log(frequency)

def get_model(args, device):
    model = NaiveBayes(
        context_window=args.context_window_length, 
        base_weight=args.base_weight
        ).to(device)
    
    config = {
        'model': args.model_type,
        'alphabet_size': model.alphabet_size,
    }
    return model, config
