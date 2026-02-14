# CleanSFT

Minimal code for Supervised Fine-Tuning (SFT) of conversational Large Language Models (LLMs) using PyTorch and Transformers.

## 📋 Description

This project implements a complete supervised fine-tuning pipeline for conversational language models.

## 🚀 Installation

```bash
# Install dependencies
pip install torch transformers datasets python-dotenv wandb tqdm
```

## 📁 Project Structure

```
grupo_estudos/
├── main.py              # Main training script
├── utils/
│   ├── __init__.py      # Args and History classes
│   └── dataset.py       # ConversationDataset
├── output/              # Trained models (auto-generated)
├── wandb/               # Weights & Biases logs
└── README.md
```

## 🎯 Basic Usage

### Training

```bash
python main.py \
  --model_name "Biatron/biatron-345m" \
  --dataset "HuggingFaceH4/ultrachat_200k" \
  --max_length 2048 \
  --per_device_batch_size 8 \
  --learning_rate 5e-6 \
  --accumulation_steps 64 \
  --epochs 2 \
  --warmup_ratio 0.1293 \
  --output_dir "./output" \
  --compile
```

## ⚙️ Training Parameters

| Parameter | Description | Default |
|-----------|-----------|--------|
| `model_name` | Name or path of base model (required) | - |
| `dataset` | HuggingFace dataset name (required) | - |
| `max_length` | Maximum sequence length | 2048 |
| `per_device_batch_size` | Batch size per device | 8 |
| `learning_rate` | Learning rate | 5e-6 |
| `accumulation_steps` | Gradient accumulation steps | 64 |
| `epochs` | Number of epochs | 2 |
| `warmup_ratio` | Warmup steps ratio | 0.1293 |
| `max_grad` | Maximum value for gradient clipping | 1.0 |
| `output_dir` | Output directory | ./output |
| `path_chat_template` | Path to custom template | None |
| `compile` | Enable torch.compile() | True |

## 🔧 Features

### ConversationDataset

Custom dataset that:
- Applies tokenizer's chat template
- Automatically masks user tokens (labels = -100)
- Trains only on assistant responses
- Supports custom templates via file

### History

Class for tracking training metrics:
- Loss per step
- Learning rate
- Gradient norm
- Automatic Weights & Biases integration
- Progress bar with tqdm

### Args

Dataclass for argument management:
- Automatic parameter validation
- JSON serialization
- Compatible with HfArgumentParser

## 📊 Dataset Format

The dataset must be in HuggingFace conversational format:

```json
{
  "messages": [
    {"role": "user", "content": "User question"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

## 🔑 Weights & Biases Configuration

Create a `.env` file with your credentials:

```bash
WANDB_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token_here
```

## 📝 Training Outputs

After training, the following files are saved in `output_dir`:

```
output/
├── config.json           # Model configuration
├── model.safetensors     # Model weights
├── tokenizer_config.json # Tokenizer configuration
├── arguments.json        # Training arguments
└── training_log.json     # Training history
```

## 🤝 Contributing

This is minimal educational code. Feel free to expand and adapt according to your needs.

## 📄 License

MIT License - free for educational and commercial use.

---

**Developed as minimal reference code for SFT on conversational LLMs.**
