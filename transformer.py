import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AutoModelForCausalLM
import transformers
from pprint import pprint

# Define the TransformerConfig class
class TransformerConfig:
    """Hyperparameters used in the Transformer architectures."""

    def __init__(self, vocab_size=256, embedding_dim=64, num_layers=4, num_heads=8, emb_init_scale=0.02, widening_factor=4):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.emb_init_scale = emb_init_scale
        self.widening_factor = widening_factor

# Function to generate the causal mask
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# Define the Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_encoding = PositionalEncoding(config.embedding_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.output_layer = nn.Linear(config.embedding_dim, config.vocab_size)
    
    def forward(self, targets, use_cache = False, cache = None, return_latent = False):# targets: (batch_size, sequence_length)
        targets = targets.transpose(0, 1)
        sequence_length = targets.shape[0]
        x = self.embedding(targets) * np.sqrt(self.config.embedding_dim)
        x = self.positional_encoding(x) # x: (sequence_length, batch_size, embedding_dim)
        attn_mask = generate_square_subsequent_mask(sequence_length).to(x.device)
        if use_cache:
            new_token_cache = []
            
            for i, layer in enumerate(self.layers):
                x = layer(x, attn_mask = None, use_cache=True)
                new_token_cache.append(x)
                if cache is not None:
                    x = torch.cat([cache[i], x], dim=0)
                    
            x = self.norm(x)
            logits = self.output_layer(x)
            
            if cache is not None:
                new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
            else:
                new_cache = torch.stack(new_token_cache, dim=0)
            logits = logits.permute(1, 0, 2)
            if return_latent:
                return logits, new_cache, x
            else:
                return logits, new_cache
        
        else:
            for layer in self.layers:
                x = layer(x, attn_mask=attn_mask)
            x = self.norm(x)
            logits = self.output_layer(x)
            logits = logits.permute(1, 0, 2)
            if return_latent:
                return logits, x
            else:
                return logits  # Return logits without applying log_softmax
    
class TransformerDecoderClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_encoding = PositionalEncoding(config.embedding_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.output_layer = nn.Linear(config.embedding_dim, 2)
    
    def forward(self, targets, use_cache = False, cache = None):# targets: (batch_size, sequence_length)
        targets = targets.transpose(0, 1)
        sequence_length = targets.shape[0]
        x = self.embedding(targets) * np.sqrt(self.config.embedding_dim)
        x = self.positional_encoding(x) # x: (sequence_length, batch_size, embedding_dim)
        attn_mask = generate_square_subsequent_mask(sequence_length).to(x.device)
        if use_cache:
            new_token_cache = []
            
            for i, layer in enumerate(self.layers):
                x = layer(x, attn_mask = None, use_cache=True)
                new_token_cache.append(x)
                if cache is not None:
                    x = torch.cat([cache[i].repeat(1,256,1), x], dim=0)
                    
            x = self.norm(x)
            logits = self.output_layer(x)
            logits = logits.permute(1, 0, 2)
            return logits, new_token_cache
        else:
            for layer in self.layers:
                x = layer(x, attn_mask=attn_mask)
            x = self.norm(x)
            logits = self.output_layer(x)
            logits = logits.permute(1, 0, 2)
            return logits  # Return logits without applying log_softmax
    

# Define PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # When d_model is odd
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        pe  = pe.permute(1, 0, 2)  # Shape (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (sequence_length, batch_size, embedding_dim)
        x = x + self.pe[:x.shape[0]]
        return x

# Define the Transformer Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=config.embedding_dim, num_heads=config.num_heads)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * config.widening_factor),
            nn.GELU(),
            nn.Linear(config.embedding_dim * config.widening_factor, config.embedding_dim)
        )
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
    
    def forward(self, x, attn_mask, use_cache = False):
        # x: (sequence_length, batch_size, embedding_dim)
        # attn_mask: (sequence_length, sequence_length)
        if use_cache:
            x_last_token = x[-1:, :, :]
            
            attn_output_last_token, _ = self.self_attn(x_last_token, x, x, attn_mask=None)
            x_last_token = self.layer_norm1(x_last_token + attn_output_last_token)
            ff_output_last_token = self.feed_forward(x_last_token)
            x_last_token = self.layer_norm2(x_last_token + ff_output_last_token)
            
            return x_last_token
        else:
            attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
            x = self.layer_norm1(x + attn_output)
            ff_output = self.feed_forward(x)
            x = self.layer_norm2(x + ff_output) 
                   
            return x

# Define a mock config class for the DecoderLayer initialization
class MockConfig:
    def __init__(self, embedding_dim=32, num_heads=4, widening_factor=2, vocab_size=256, num_layers=4):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.widening_factor = widening_factor
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
class myGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        
    def forward(self, x):
        x = x.transpose(0, 1)
        input = transformers.tokenization_utils_base.BatchEncoding().to(x.device)
        input['input_ids'] = x
        input['attention_mask'] = torch.ones_like(x)
        
        logits = self.gpt2(**input).logits
        logits = logits[:,:,:256]
        
        logits = logits.permute(1, 0, 2)
        return logits
    
class myLamma3(nn.Module):
    def __init__(self):
        super().__init__()
        self.lamma3 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        
    def forward(self, x):
        x = x.transpose(0, 1)
        input = transformers.tokenization_utils_base.BatchEncoding().to(x.device)
        input['input_ids'] = x
        input['attention_mask'] = torch.ones_like(x)
        
        logits = self.lamma3(**input).logits
        logits = logits[:,:,:256]
        
        logits = logits.permute(1, 0, 2)
        return logits
    
def get_model(args, load_model, device):
    if load_model is None:
        config = TransformerConfig(embedding_dim=args.embedding_dim, num_heads=args.num_heads, num_layers=args.num_layers)
        pprint(config)
        model = TransformerDecoder(config)
    elif load_model == "gpt2":
        model = myGPT2()
        config = None
    elif load_model == "lamma3":
        model = myLamma3()
        config = None
    else:
        checkpoint = torch.load(
                load_model,
                map_location="cpu",
                weights_only=False,  # legacy checkpoints require full pickle load
            )
        
        config = checkpoint['config']
        pprint(config)
        model = TransformerDecoder(config)
        model.load_state_dict(checkpoint['model'])
        
        print("=" * 40)
        print('Model parameters loaded.')
        print("In-distribution validation bpd: ", checkpoint['validation_bpd'])
        print("=" * 40)
    
    model = nn.DataParallel(model)  # Data parallelism
    model = model.to(device)
    
    return model, config

# Test block for DecoderLayer
def test_decoder_layer():
    # Initialize the mock config
    config = MockConfig()

    # Create a DecoderLayer instance
    decoder_layer = DecoderLayer(config)
    decoder_layer.eval()

    # Sample inputs
    batch_size = 4
    sequence_length = 10
    embedding_dim = config.embedding_dim

    # Random input tensor simulating (batch_size, sequence_length, embedding_dim)
    x = torch.randn(sequence_length, batch_size, embedding_dim)

    # Attention mask (sequence_length, sequence_length)
    attn_mask = generate_square_subsequent_mask(sequence_length)
    with torch.no_grad():
        # Forward pass
        all_output = decoder_layer(x, attn_mask)
        assert all_output.shape == (sequence_length, batch_size, embedding_dim)
        print("is same", torch.all(all_output == decoder_layer(x, attn_mask)))
        for i in range(x.shape[0]):
            current_x = x[:i + 1, :, :]
            output = decoder_layer(current_x, None, use_cache=True)
            assert output.shape == (1, batch_size, embedding_dim)
            ground_truth_output = all_output[i, :, :].unsqueeze(0)
            diff = output - ground_truth_output
            print(f'position {i + 1}', f'is the same {torch.allclose(diff, torch.tensor(0.))}', diff.norm().item()) 
    
    print("Test passed.")
    
def test_decoder():
    config = MockConfig()
    decoder = TransformerDecoder(config)
    batch_size = 4
    sequence_length = 3096
    targets = torch.randint(0, config.vocab_size, (sequence_length, batch_size))
    with torch.no_grad():
        all_logits = decoder(targets)
        assert all_logits.shape == (sequence_length, batch_size, config.vocab_size)
        
        cache = None
        for i in range(sequence_length):
            current_targets = targets[:i + 1, :]
            logits, cache = decoder(current_targets, use_cache=True, cache=cache)
            assert logits.shape == (i + 1, batch_size, config.vocab_size)
        
        for j in range(sequence_length):
            ground_truth_logits = all_logits[j, :, :]
            target_logits = logits[j, :, :]
            diff = target_logits - ground_truth_logits
            print(f'position {j + 1}', f'is the same {torch.allclose(diff, torch.tensor(0.))}', diff.norm().item())
    
def test_time(config, batch_size, sequence_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = TransformerDecoder(config)
    decoder.to(device)
    targets = torch.randint(0, config.vocab_size, (sequence_length, batch_size))
    targets = targets.to(device)
    with torch.no_grad():
        t_0 = time.time()
        for i in tqdm(range(sequence_length)):
            current_targets = targets[:i + 1, :]
            logits = decoder(current_targets)
        t_1 = time.time()
        print("inference without cache finished")
        
        cache = None
        for i in range(sequence_length):
            current_targets = targets[:i + 1, :]
            logits, cache = decoder(current_targets, use_cache=True, cache=cache)
            assert logits.shape == (i + 1, batch_size, config.vocab_size)
        t_2 = time.time()
        print("inference with cache finished")
        print('dim', config.embedding_dim)
        print("Time without cache", t_1 - t_0)
        print("Time with cache", t_2 - t_1)
        print("Speedup", (t_1 - t_0) / (t_2 - t_1))
        print("=" * 20)
        return t_1 - t_0, t_2 - t_1
    
def test_muti_config(batch_size = 32, sequence_length = 2048):
    embedding_dims = [64, 128, 256]
    for embedding_dim in embedding_dims:
        config = TransformerConfig(embedding_dim=embedding_dim, vocab_size=256)
        test_time(config, batch_size, sequence_length)
        
def test_decoder_classifier():
    import wandb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=f'KV_cache_Classifier', name=f'{device}')
    config = TransformerConfig(embedding_dim=64, vocab_size=256)
    decoderclassifier = TransformerDecoderClassifier(config)
    decoderclassifier.to(device)
    batch_size = 1
    sequence_length = 2048
    batch = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
    batch = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long), batch], dim=1)
    batch = batch.to(device)
    print(batch)
    
    def update_cache(cache, new_token_cache, batch):
        new_token_cache = torch.stack(new_token_cache, dim=0)
        batch = batch.view(1, 1, 1, 1)
        batch = batch.repeat(new_token_cache.shape[0], new_token_cache.shape[1], 1, new_token_cache.shape[3])
        new_token_cache = new_token_cache.gather(2, batch) #Shape (num_layers, 1, batch_size, dim)
        if cache is not None:
            new_cache = torch.cat([cache, new_token_cache], dim=1)
        else:
            new_cache = new_token_cache
            
        return new_cache
    
    t1_sum = 0
    t2_sum = 0
    with torch.no_grad():
        all_alphabet = torch.arange(256).unsqueeze(-1).to(batch.device) # shape: (256, 1)
        
        cache = None
        new_token_cache = None
        for i in tqdm(range(batch.shape[-1])):
            current_batch = batch[:, :i]
            current_batch = current_batch.repeat(256, 1)
            current_batch = torch.cat([current_batch, all_alphabet], dim=-1) # shape: (256, idx + 1）
            t_0 = time.time()
            if new_token_cache is not None:
                cache = update_cache(cache, new_token_cache, batch[:,i - 1])
                
            current_logits, new_token_cache = decoderclassifier(current_batch, use_cache=True, cache=cache) # shape: (256, i + 1, 2)
            t_1 = time.time()
            original_logits = decoderclassifier(current_batch) # shape: (256, i + 1, 2)
            t_2 = time.time()
            diff = current_logits[:,-1,:] - original_logits[:,-1,:]
            print(f'position {i}', f'is the same {torch.allclose(diff, torch.tensor(0.))}', diff.norm().item())
            t1_sum += t_1 - t_0
            t2_sum += t_2 - t_1
            acceleration = t2_sum / t1_sum
            print("Time with cache", t1_sum, "Time without cache", t2_sum, "Acceleration", acceleration)
            wandb.log({
                       "Time with cache": t1_sum, 
                       "Time without cache": t2_sum, 
                       "Acceleration": acceleration,
                       "Diff": diff.norm().item(),
                       "is the same": torch.allclose(diff, torch.tensor(0.))
                       })
            
def test_mygpt2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = myGPT2()
    decoder.to(device)
    targets = torch.randint(0, 256, (1024, 8))
    targets = targets.to(device)
    logits = decoder(targets)

# If the script is executed directly, run the test
if __name__ == "__main__":
    # torch.set_default_dtype(torch.float64) 
    # test_decoder_classifier()  
    test_mygpt2()
    # test_decoder_layer()
    # test_decoder()
    # test_muti_config(sequence_length = 2048)
    # test_muti_config(sequence_length = 3072)
