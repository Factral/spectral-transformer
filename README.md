# spectral-transformer

dataset:
https://data.csiro.au/collection/csiro%3A55630v4

install requirements:

```bash
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

prepare dataset:
```bash
python3 prepare_dataset.py --dir /path/to/extracted/dataset --split train
python3 prepare_dataset.py --dir /path/to/extracted/dataset --split test
python3 prepare_dataset.py --dir /path/to/extracted/dataset --split val
```

train:
```bash
sh train.sh
```
