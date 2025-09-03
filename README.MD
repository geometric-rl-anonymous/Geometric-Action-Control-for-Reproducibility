# GAC: Geometric Action Control

Official evaluation code for "Beyond Distributions: Geometric Action Control for Continuous Reinforcement Learning"

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- Gymnasium

### Installation

```bash
pip install torch gymnasium numpy tqdm
```

## Pretrained Models

### Evaluate Pretrained Models

```bash
# Evaluate specific environment
python eval_gac.py --checkpoint gac_pretrain_model/HalfCheetah-v4_step_500000.pt --episodes 10
```

## Pretrained Models

| Environment    | Steps | Expected Return |
| -------------- | ----- | --------------- |
| HalfCheetah-v4 | 500k  | 12,617 ± 168    |
| Ant-v4         | 500k  | 5,759 ± 68      |
| Walker2d-v4    | 1M    | 5,889 ± 47      |
| Humanoid-v4    | 1M    | 6,591 ± 53      |

*Note: Returns are from deterministic evaluation over 10 episodes.*

## Files

- `eval_gac.py`: Evaluation script
- `gac_core_protected.so`: Core GAC implementation (C++ extension)
- `pretrained_models/`: Checkpoints for reproduction

## License

MIT

