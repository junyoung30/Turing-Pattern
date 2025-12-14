# 🚀 사용 방법 (How to Use)

이 프로젝트는 시뮬레이션을 이용한 데이터 생성과 CNN 학습의 전체 파이프라인을 제공합니다.
파이썬 3.7.2 환경에서 개발 및 실험되었습니다.

---

### 1) Install dependencies
필요한 모든 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

---

### 2) Generate patterns (test)
선택한 파라미터(Dv, k1)에 대해 2D 패턴을 시뮬레이션합니다. 
데이터셋을 만들기 전에 해당 파라미터가 어떤 패턴을 생성하는지 확인하는데 활용합니다.

```bash
python generate_pattern.py --Dv 0.01 --k1 5 --seed 1004
```

---

### 3) Build Dataset
여러 파라미터 조합과 seeds를 이용해 학습용 데이터셋을 생성합니다.

```bash
python generate_dataset.py --points 0.01,5,Sdot 0.04,1,Ldot --seeds 1 2 3 4 5
```

---

### 4) Run training
생성된 데이터셋을 기반으로 CNN 모델을 학습합니다.

```bash
python train.py --num_blocks 1 --num_conv 6 --aug 0 --weight_seed 2025
```
