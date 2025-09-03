# GAC: Geometric Action Control

Official evaluation code for **"Beyond Distributions: Geometric Action Control for Continuous Reinforcement Learning"**.

This repo provides pretrained models and a minimal evaluation script for reproducibility.

---

## 🛠️ Quick Start

### Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- Gymnasium

### Install dependencies

```bash
pip install torch gymnasium numpy tqdm
```



------

## 🚀 Evaluate Pretrained Models

Run the evaluation script on a specific environment:

```bash
python eval_gac.py --checkpoint gac_pretrain_model/HalfCheetah-v4_step_500000.pt --episodes 10
```

> Default: deterministic evaluation over 10 episodes.

------

## 📦 Pretrained Models

| Environment    | Steps | Return (mean ± std) |
| -------------- | ----- | ------------------- |
| HalfCheetah-v4 | 500k  | 12,617 ± 168        |
| Ant-v4         | 500k  | 5,759 ± 68          |
| Walker2d-v4    | 1M    | 5,889 ± 47          |
| Humanoid-v4    | 1M    | 6,591 ± 53          |

> 📍 Models were trained using default GAC settings: fixed radius $r=2.5$ and adaptive $\kappa$.

------

## 📁 Repository Structure

- `eval_gac.py` — Evaluation script
- `gac.so` — GAC core (C++ extension)
- `gac_pretrain_model/` — Pretrained checkpoints
- `requirements.txt` — Dependency list

------

## 📄 License

MIT License

------



