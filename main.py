import argparse

import torch
import json
from collections import defaultdict
from models.llama import LlamaConfig, LlamaDecoderLayer, LlamaRotaryEmbedding

def test_llama_decoder_layer(args):
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config_dict['_attn_implementation'] = args.attn_implementation

    config = LlamaConfig(**config_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Profiling will be on CPU and may not be accurate.")
        return

    layer_idx = 0
    decoder_layer = LlamaDecoderLayer(config, layer_idx).to(device)
    
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        raise NotImplementedError
    
    decoder_layer = decoder_layer.to(dtype)
    decoder_layer.eval()
    
    print(f"Model dtype: {next(decoder_layer.parameters()).dtype}")
    
    batch_size = args.batch_size
    seq_length = args.seq_length  
    
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size, device=device, dtype=dtype)
    print(hidden_states.dtype)
    
    causal_mask = torch.triu(torch.full((seq_length, seq_length), -torch.finfo(torch.float32).min, device=device), diagonal=1)
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Rotary Position Embeddings
    rotary_emb = LlamaRotaryEmbedding(config, device=device)
    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)
    position_embeddings = rotary_emb(hidden_states, position_ids)

    timings = defaultdict(float)
    num_runs = 10

    # warm up
    print("\nWarming up GPU...")
    with torch.no_grad():
        _ = decoder_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            timings=None 
        )

    print(f"Profiling forward pass over {num_runs} runs...")
    with torch.no_grad():
        for _ in range(num_runs):
            _ = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                timings=timings 
            )

    total_time = sum(timings.values())
    print("\n--- LlamaDecoderLayer Performance Profile ---")
    print(f"Total Average Time: {total_time / num_runs:.4f} ms")
    print("---------------------------------------------")

    sorted_timings = sorted(timings.items(), key=lambda item: item[1], reverse=True)
    for name, time_ms in sorted_timings:
        avg_time = time_ms / num_runs
        percentage = (time_ms / total_time) * 100
        print(f"{name:<30} | Avg Time: {avg_time:>8.4f} ms | Percentage: {percentage:>5.2f}%")
    print("---------------------------------------------")
    
    with open(f'./results/{args.attn_implementation}_{args.seq_length }.txt', 'w') as f:
        f.write(f"\n--- LlamaDecoderLayer Performance Profile ---\n")
        f.write(f"Total Average Time: {total_time / num_runs:.4f} ms\n")
        f.write("---------------------------------------------\n")
        for name, time_ms in sorted_timings:
            avg_time = time_ms / num_runs
            percentage = (time_ms / total_time) * 100
            f.write(f"{name:<30} | Avg Time: {avg_time:>8.4f} ms | Percentage: {percentage:>5.2f}%\n")
        f.write("---------------------------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=2048)
    parser.add_argument('--config_path', type=str, default='./config/llama2-7b.json')
    parser.add_argument('--dtype', type=str, default='float16', choices=['bfloat16', 'float16'])
    parser.add_argument('--attn_implementation', type=str, default='eager', choices=['eager', 'triton', 'triton_wo_softmax'])
    args = parser.parse_args()
    test_llama_decoder_layer(args)