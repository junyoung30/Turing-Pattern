사용법

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Generate patterns (test)
```bash
python generate_patterns.py --Dv 0.01 --k1 5 --seed 1004
```

### 3) Build Dataset
```bash
python generate_dataset.py \
    --points 0.01,5,Sdot 0.04,1,Ldot \
    --seeds 1 2 3 4 5 \
```

### 4) Run training
```bash
python train.py 
```
