Here is the complete `README.md` file. You can copy this entire block and save it directly as a `.md` file in your repository.

```markdown
# STLEnc: Neural Signal Temporal Logic Encoder

STLEnc is a Transformer-based encoder designed to map **Signal Temporal Logic (STL)** formulae into a continuous 1024-dimensional latent embedding space. 

This project implements a neural approximation of the kernel-based framework introduced by **Gallo et al.** in [*"A Kernel-Based Approach to Signal Temporal Logic"* (2020)](https://arxiv.org/abs/2009.05484). Instead of calculating recursive kernels against an anchor set, this model learns to project STL syntax directly into a semantically meaningful vector space.



---

## 1. Project Structure
- `train.py`: Main script for training the backbone using MSE loss against kernel targets.
- `requirements.txt`: List of necessary Python packages.
- `.gitignore`: Rules to keep the repository clean of temporary files and cache.
- `src/`: Custom architecture and tokenizer logic (mirrored from Hugging Face).

---

## 2. Setup and Installation

This project requires **Python 3.11**. Follow these steps to set up your local environment using `venv`.

### Create and Activate the Environment
```bash
# Create the virtual environment
python3.11 -m venv .venv

# Activate it
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### Authentication

Log in to Hugging Face (to access model/datasets) and Weights & Biases (for experiment tracking):

```bash
huggingface-cli login
wandb login

```

---

## 3. Training

The model is trained to approximate the STL kernel by minimizing the Mean Squared Error (MSE) between the Transformer's output and the pre-computed embeddings from the [saracandu/stl_formulae](https://www.google.com/search?q=https://huggingface.co/datasets/saracandu/stl_formulae) dataset.

To start the training with **Weights & Biases** integration:

```bash
python train.py

```

During training, you can monitor the **Average Cosine Similarity** on your WandB dashboard. This metric indicates how well the Transformer is approximating the Gallo et al. kernel (values closer to 1.0 are better).

---

## 4. Usage

You can load the latest trained backbone directly from the Hugging Face Hub. Note that `trust_remote_code=True` is required for the custom STL tokenizer and model classes.

```python
import torch
from transformers import AutoModel, AutoTokenizer

repo_id = "saracandu/stlenc"

# Load Backbone
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Encode a formula
formula = "always[0, 10] (x > 0) and eventually[5, 20] (y < -1)"
inputs = tokenizer(formula, return_tensors="pt")

with torch.no_grad():
    embeddings = model(**inputs)

print(embeddings.shape) # Output: torch.Size([1, 1024])

```

---

## 5. Scientific Background

The core objective is to approximate the STL kernel mapping . This neural approach offers:

* **Efficiency**: Faster embedding generation than recursive kernel methods.
* **Scalability**: Capable of handling large-scale datasets for downstream tasks like robustness prediction or formula classification.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{gallo2020kernel,
  title={A Kernel-Based Approach to Signal Temporal Logic}, 
  author={Luca Gallo and Alberto Padoan and Luca Ballotta and Luca Schenato and Maria Elena Valcher},
  year={2020},
  journal={arXiv preprint arXiv:2009.05484}
}

@software{saracandu_stlenc_2025,
  author = {Sara Candussio},
  title = {STL Encoder: A Neural Backbone for Signal Temporal Logic},
  year = {2025},
  url = {[https://huggingface.co/saracandu/stlenc](https://huggingface.co/saracandu/stlenc)}
}

```
