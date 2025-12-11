# 사용 방법 (How to Use)

이 프로젝트는 시뮬레이션을 이용한 데이터 생성과 CNN 학습의 전체 파이프라인을 제공합니다.

---

### 1) Install dependencies
필요한 모든 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

---

### 2) Generate patterns (test)
```bash
python generate_patterns.py --Dv 0.01 --k1 5 --seed 1004
```

### 3) Build Dataset
```bash
python generate_dataset.py --points 0.01,5,Sdot 0.04,1,Ldot --seeds 1 2 3 4 5
```

### 4) Run training
```bash
python train.py --num_blocks 1 --num_conv 6 --aug 0 --weight_seed 2025
```
