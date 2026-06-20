## How to activate the environment

### For conda:
```bash
conda env create -f environment.yml
conda activate thesis2026
```

### For venv:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```