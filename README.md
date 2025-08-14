# LLM Decoder Layer Profiling

A performance profiling tool for LLM decoder layers with different attention implementations.

## Project Structure (only support llama decoder layer now)

```
non-linear-profile/
├── main.py                 # Main profiling script
├── scripts/profile.sh      # Batch profiling runner
├── requirements.txt        # Python dependencies
├── config/llama2-7b.json  # Model configuration
├── models/                 # LLaMA model implementation
├── triton_kernel/          # Custom Triton kernels
├── results/                # Profiling output files
└── timing_utils.py         # Timing utilities
```

## Environment Setup

```bash
# Create conda environment
conda create -n profile python=3.9
conda activate profile

# Install dependencies
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Usage

### Single Run
```bash
python main.py --seq_length 2048 --attn_implementation triton
```

### Batch Profiling
```bash
sh scripts/profile.sh
```

Tests combinations of:
- **Sequence lengths**: 1024, 2048, 4096, 8192, 16384
- **Attention implementations**: eager, triton, triton_wo_softmax

## Output

Results are saved to `results/` directory with timing breakdowns for each component:
- Attention mechanisms (QKV projections, softmax, matmul)
- MLP layers (gate, up, down projections)
- Layer normalization
- RoPE embeddings

## Parameters

- `--seq_length`: Input sequence length (default: 2048)
- `--attn_implementation`: Attention type (eager/triton/triton_wo_softmax)
- `--batch_size`: Batch size (default: 1) 
- `--dtype`: Data type (float16/bfloat16, default: float16)
- `--config_path`: Model config file (default: ./config/llama2-7b.json)
