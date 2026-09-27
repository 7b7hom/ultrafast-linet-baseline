# 실험 상세 안내

프로젝트 개요와 현재 진행 상태는 [README](../README.md), 최종 수치는 [최종 보고서](../reports/FINAL_RESULTS.md)를 참고하세요. 이 문서에는 기존 README의 세부 설정과 실행 절차를 모았습니다. 아래 명령은 저장소 루트에서 실행합니다.

## 연구 질문과 실험 범위

> 여러 세기의 잡음으로 학습하면 UltraFast-LiNET-Max의 구조를 바꾸지 않고도 잡음이 추가된 저조도 사진의 보정 품질을 높일 수 있는가? 그 과정에서 원래 입력의 품질과 CPU 실행 비용은 어떻게 달라지는가?

예정된 비교는 다음과 같습니다. **현재 학습 스크립트는 A만 구현합니다.**

| 모델 | 학습 입력 | 확인할 효과 |
|---|---|---|
| A | 원본 저조도 입력 100% | 기준 성능 |
| B | 원본 50% + Gaussian σ=0.03 입력 50% | 잡음 증강 자체의 효과 |
| C | 원본 50% + σ∈{0.01, 0.03, 0.05} 중 균등 선택한 입력 50% | 여러 잡음 세기로 학습한 효과 |

픽셀 범위는 `[0,1]`이며 증강식은 `clamp(x + noise, 0, 1)`입니다. 정상 조도 정답 `y`에는 잡음을 넣지 않습니다. 후속 Gaussian 평가의 시작안은 σ∈{0, 0.01, 0.03, 0.05, 0.07}입니다. σ=0.07은 제안된 학습 범위 밖의 조건입니다. 범위는 검증 데이터에서 확정해야 하며, 이 저장소의 현재 결과가 잡음 강건성을 입증하는 것은 아닙니다.

LOL-v1의 원래 저조도 사진에도 촬영 잡음이 있을 수 있으므로 σ=0 조건은 **‘추가 합성 잡음 없음’**으로 표현합니다. 합성 Gaussian 잡음의 효과를 실제 센서 잡음 전체에 대한 일반화로 확대하지 않습니다. 동일 구조의 A/B/C에서 기대하는 성과는 **추론 구조를 늘리지 않은 품질 개선**이며, 속도 향상은 별도로 측정해야 하는 주장입니다.

## 원본 출처와 고정 버전

| 항목 | 고정 값 |
|---|---|
| 공식 저장소 | [YuhanChen2024/UltraFast-LiNET](https://github.com/YuhanChen2024/UltraFast-LiNET) |
| 공식 코드 commit | [`12e8c79566235fecc952d9a148011cf5fbb2c8ba`](https://github.com/YuhanChen2024/UltraFast-LiNET/tree/12e8c79566235fecc952d9a148011cf5fbb2c8ba) |
| 참고 논문 | [UltraFast-LiNET, arXiv:2512.02965v2](https://arxiv.org/html/2512.02965v2) |
| 모델 | `ultrafast_linet_max()`, κ=5, dia=(1,2,3,4,5), gate=True, bottleneck_mode="down" |
| 파라미터 | 전체 180개 / 학습 가능 180개 |
| FP32 파라미터 텐서 크기 | 720 bytes |
| 공식 checkpoint SHA-256 | `6f8b28b3b9aef9efc00adcaba84ed3a30237ed8335534994379a9178d9252ae1` |

[`vendor/`](../vendor/)의 Python 파일은 고정한 공식 코드 원본입니다. 파일별 SHA-256과 공개 checkpoint의 용도는 [provenance.json](../provenance.json)에 기록합니다. 공개 checkpoint는 추론 동작 확인에만 사용하며, **A 학습의 초기값으로 로드하지 않습니다.** 공개 checkpoint의 `epoch=120` 메타데이터만으로 논문의 360 epoch 학습 이력을 복구할 수는 없습니다.

### 논문과 공개 구현 사이의 차이

이 프로젝트는 **공개 checkpoint와 호환되는 공식 구현**을 기준으로 삼습니다. 논문의 수식과 구현을 모두 동일하게 재현했다고 주장하지 않습니다.

- 공개 `model.py`의 디코더 연결은 `d1←e3`, `d2←b`, `d3←d1`입니다. 최종 출력 `d3`의 경로는 bottleneck `b`를 직접 통과하지 않습니다. 논문의 순차 연결 `d1←b`, `d2←d1`, `d3←d2`로 바꾸지 않았습니다.
- `bottleneck_mode="down"`으로 bottleneck에서도 해상도를 줄입니다. 공식 코드의 설명에서 Eq. (18)을 문자 그대로 따르는 대안은 `"same"`입니다.
- 논문의 dia 표기 0…4와 공개 코드 기본값 1…5를 구분합니다. 여기서는 공개 코드 기본값을 유지합니다.
- encoder, bottleneck, decoder용 MSRB 세 모듈을 각 스케일에서 공유합니다. 같은 모듈을 반복 호출하며 스케일별로 독립 파라미터를 추가하지 않습니다.
- 공식 `train.py` 예시의 시험 세트 기반 checkpoint 선택을 사용하지 않습니다. 별도 `scripts/train_a.py`가 검증 50쌍만으로 선택합니다.
- 원래 학습 485쌍 중 50쌍을 검증에 배정하므로 실제 학습은 435쌍입니다. 논문의 485쌍 학습 결과와 조건이 같지 않습니다.

모델 연결을 수정하는 실험은 입력 잡음 증강만 바꾸는 A/B/C 비교와 분리해야 합니다.

## 데이터와 고정 분할

[LOL 공식 배포 페이지](https://daooshee.github.io/BMVC2018website/)에서 제공하는 LOL-v1을 사용합니다. 데이터 원본은 저장소에 포함하지 않으며, 직접 내려받아 다음 구조로 배치합니다.

```text
data/LOL/
├── our485/
│   ├── low/                 # 원래 저조도 학습 이미지 485개
│   └── high/                # 대응 정상 조도 정답 485개
└── eval15/
    ├── low/                 # 공식 시험 입력 15개
    └── high/                # 공식 시험 정답 15개
```

| 분할 | 쌍 수 | 사용 목적 |
|---|---:|---|
| train | 435 | 모델 가중치 업데이트 |
| validation | 50 | 개발 설정 확인·checkpoint 선택 |
| test | 15 | 전체 학습 및 선택 완료 후 최종 평가 |

원래 학습 485쌍에서 **정상 조도 파일이 바이트 단위로 같은 66개 그룹, 총 152쌍**을 발견했습니다. 동일 low 또는 high 파일의 SHA-256을 공유하는 쌍을 연결 성분으로 묶고, seed=42의 그룹 순서와 결정적인 부분합 선택으로 검증 세트를 정확히 50쌍으로 정했습니다. 연결된 중복 그룹은 train/validation 양쪽으로 나뉘지 않습니다.

이는 **바이트가 같은 중복의 누출 방지**입니다. 같은 장면의 다른 촬영이나 시각적으로 유사한 장면까지 모두 독립이라고 보장하지 않습니다. 단순 파일명 무작위 435/50 분할과 목록이 다르므로 비교 실험에서는 제공된 manifest를 그대로 사용합니다.

[manifests/](../manifests/)에는 각 쌍의 데이터 루트 기준 상대 경로와 low/high SHA-256이 있습니다. 고정 manifest 파일의 해시는 다음과 같습니다.

| 파일 | SHA-256 |
|---|---|
| `train.json` | `03209fd4ef17be7794a675a52b54087cd98c6aa36230eaf23692022e9e2c54ba` |
| `validation.json` | `5c835462afe878fc0acc181c31c0ebdaac337203bc686101e3ec8c5915e4d086` |
| `test.json` | `653c1078718191477587e0915ba77e9045197c27471a16e33b673dfb5afc4df1` |

분할 생성 단계에서는 시험 파일의 이름과 해시만 확인했습니다. 이후 사용자가 요청한 최종 A 평가는 전체 학습 완료 후 `evaluate.py`로 수행합니다. manifest에 기록된 `test_pixels_decoded=false`는 **분할 생성 당시**의 상태이며, 이후 최종 평가의 미실행을 뜻하는 영구적인 표시는 아닙니다. 실제 평가 여부는 최종 평가 산출물로 확인합니다.

압축파일 출처와 해시는 [data_provenance.json](../data_provenance.json)에 있습니다. 사용한 `LOLdataset.zip`의 SHA-256은 `47d85314b7927470cd48c97e5c1d6c896a56d6217c8e508d0736aff2d91aadcc`입니다.

## 설치

이하 명령은 저장소 루트에서 실행합니다. 검증 환경은 **Python 3.12.6, PyTorch 2.14.0, macOS 26.6.2 ARM64, Apple M2 Pro**입니다. 설치 버전은 [requirements.lock.txt](../requirements.lock.txt)에 고정했습니다. 다른 운영체제·가속기에서의 수치 일치와 실행은 별도 검증이 필요합니다.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

`uv`를 이미 사용한다면 다음 명령으로 같은 환경을 만들 수 있습니다.

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock.txt
```

접근 권한이 있는 계정으로 이 저장소를 복제한 뒤 위 명령을 실행합니다. 저장소의 공개 범위와 관계없이 데이터는 직접 내려받아 준비합니다.

기본 실행 장치는 CPU이며 CUDA가 필요하지 않습니다. 오래된 THOP의 `distutils` 의존성을 위해 `setuptools==80.9.0`도 고정하고 THOP를 불러오기 전에 호환 모듈을 활성화합니다.

## 실행 방법

### 1. 데이터 내려받기와 분할 확인

공식 페이지의 Google Drive 링크를 사용하거나 아래 명령을 실행합니다. Google Drive 다운로드 제한이 발생하면 공식 페이지에서 직접 받은 파일을 같은 위치에 두면 됩니다.

```bash
mkdir -p data/LOL
.venv/bin/gdown 157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB -O data/LOLdataset.zip
.venv/bin/python -m zipfile -e data/LOLdataset.zip data/LOL
.venv/bin/python scripts/prepare_data.py \
  --data-root data/LOL --output manifests --seed 42
```

압축 해제 후 `data/LOL/our485`와 `data/LOL/eval15`가 직접 존재하는지 확인합니다. 추가 상위 폴더가 생긴 경우 `--data-root`를 이 두 폴더가 있는 위치로 지정합니다.

`prepare_data.py`는 low/high 파일명·개수·해시, 원래 학습 485쌍의 RGB 모드·대응 크기·무결성을 확인합니다. 기존 manifest와 결과가 다르면 덮어쓰기를 거부합니다. 학습 시작 시에도 train/validation 파일 해시를 다시 확인하므로 분할 이후 파일을 변경하면 오류가 납니다.

### 2. 공개 checkpoint의 CPU 추론 확인

```bash
.venv/bin/python scripts/reproduce.py \
  --input data/LOL/our485/low/10.png \
  --target data/LOL/our485/high/10.png \
  --weights weights/official_max.pkl \
  --output reports/reproduction \
  --threads 1 --warmup 10 --repeats 50
```

이 명령은 `input.png`, `enhanced.png`, `target.png`, `comparison.png`, `metrics.json`을 로컬에 생성합니다. 이 공개 checkpoint의 진단 결과와 README에 실은 A 모델의 테스트 예시는 서로 다른 결과입니다.

`10.png`는 공식 원래 학습 세트의 이미지입니다. 공개 checkpoint가 이미 학습에 사용했을 수 있어 이 이미지의 PSNR·SSIM은 독립 시험 성능이 아닙니다. 새 A 모델의 결과와도 구분합니다.

### 3. 짧은 학습 동작 확인: 선택 사항

```bash
.venv/bin/python scripts/train_a.py \
  --data-root data/LOL \
  --config configs/baseline_a.json \
  --manifests manifests \
  --output runs/a_pilot_seed42 \
  --epochs 1 --pilot
```

`--pilot`는 checkpoint와 요약에 파일럿임을 기록합니다. 파일럿 실행을 완료된 360 epoch 기준선으로 간주하지 않으며, 전체 학습은 별도 폴더에서 처음부터 시작합니다.

### 4. A 모델 360 epoch 전체 학습

```bash
.venv/bin/python scripts/train_a.py \
  --data-root data/LOL \
  --config configs/baseline_a.json \
  --manifests manifests \
  --output runs/a_team_seed42
```

위 예시는 저장소에 포함된 기존 결과와 구분하기 위해 `runs/a_team_seed42`에 저장합니다. 새 학습 실행은 매번 새 출력 폴더를 사용해야 합니다. 기존 폴더에서는 명시적인 재개 옵션 없이 덮어쓰기를 거부합니다. checkpoint는 원자적으로 저장하며 중단된 실행은 마지막 완료 epoch에서 재개할 수 있습니다. 장시간 실행을 위한 전원·세션 상태를 준비합니다. `--device cuda` 또는 `--device mps` 옵션은 있지만, 현재 확인한 학습 환경은 CPU입니다. 다른 장치에서는 지원 연산·결정성과 전체 설정을 다시 기록해야 합니다.

표준 출력과 `runs/a_team_seed42/log.csv`에서 epoch별 loss, 검증 PSNR·SSIM, 학습률, 시간, 처리 이미지 수를 확인할 수 있습니다. `summary.json`은 학습 루프가 정상 종료되거나 `--stop-after-epoch`로 계획적으로 멈춘 뒤 생성됩니다. 파일이 존재한다는 사실만으로 360 epoch 완료를 뜻하지 않습니다. 전체 A 실행에서는 `completed_epochs=360`, `pilot=false`, `full_baseline_complete=true`를 확인합니다.

학습이 중단된 경우 같은 데이터·manifest·설정으로 마지막 checkpoint를 지정합니다. 재개 시 설정 일치 여부를 확인하며, 이미 처리 중이던 미완료 epoch는 다시 실행합니다.

```bash
.venv/bin/python scripts/train_a.py \
  --data-root data/LOL \
  --config configs/baseline_a.json \
  --manifests manifests \
  --output runs/a_team_seed42 \
  --resume runs/a_team_seed42/last.pt
```

### 5. 검증으로 선택한 모델의 최종 시험 평가

전체 학습이 완료된 후 다음 명령을 실행합니다.

```bash
.venv/bin/python scripts/evaluate.py \
  --data-root data/LOL \
  --run-dir runs/a_team_seed42 \
  --manifests manifests \
  --output runs/a_team_eval_seed42
```

평가 스크립트는 완료된 전체 학습 기록을 확인한 뒤 `best.pt`를 공식 시험 15쌍에서 평가합니다. 1 epoch 파일럿이나 미완료 실행을 최종 기준선으로 채점하지 않도록 완료 조건을 검사합니다. 시험 데이터는 학습이나 checkpoint 선택에 사용하지 않습니다. `metrics.json`에는 집계·이미지별 점수와 checkpoint·분할 검증 기록을, `per_image.csv`와 `aggregate.csv`에는 표 형태의 결과를 저장합니다. 기본적으로 이미지를 저장하지 않으며 `--save-images`를 추가하면 보정 출력만 로컬에 저장합니다. 원본 입력·정답은 복사하지 않습니다. 평가 옵션은 `--help`로 확인할 수 있습니다.

```bash
.venv/bin/python scripts/evaluate.py --help
```

아래는 **저장소에 포함된 기존 A 결과 문서를 다시 생성하는 명령**입니다. 완료된 학습·평가 기록을 확인한 뒤 README와 최종 보고서를 갱신합니다. 위에서 새로 실행한 `runs/a_team_*` 결과는 해당 폴더의 지표 파일에서 확인하세요. 현재 문서 생성기는 결과 링크를 기존 `reports/` 경로로 고정하므로 새 실행 경로를 넣지 않습니다.

```bash
.venv/bin/python scripts/summarize_results.py \
  --run-dir reports/a_full_seed42 \
  --evaluation-dir reports/a_final_seed42
```

최종 A 시험 점수를 확인한 후에는 **그 결과를 보고 B/C의 잡음 범위·학습률·선택 기준을 조정하지 않습니다.** 후속 설정 결정은 검증 50쌍에서 진행하고 시험 15쌍은 고정된 프로토콜의 비교 보고에 사용합니다. 이미 확인한 시험 결과를 개발 과정에 참고했다면 그 사실과 한계를 보고해야 합니다.

### 6. 자동 검증

```bash
.venv/bin/python -m pytest tests -q
```

초기 재현 시점에는 테스트 10개가 통과했으며, 재개·최종 평가 검증을 추가한 현재 테스트 26개가 모두 통과했습니다. 공식 소스 해시, checkpoint 호환성, 홀수 해상도 출력, shift 연산, 유한한 gradient, 시드 초기화, 지표, 대응 crop, 잘못된 쌍 거부, 중복 분리 및 연산량 집계를 확인합니다. 새로운 평가 코드의 검증 결과는 최신 실행 기록을 따릅니다. 공식 CLI와 새 추론 스크립트가 저장한 PNG의 최대 uint8 픽셀 오차는 0이었습니다([비교 기록](../reports/official_cli_parity.json)).

## 학습·checkpoint 선택 규칙

기본 설정 파일은 [configs/baseline_a.json](../configs/baseline_a.json)입니다.

| 설정 | 값 |
|---|---|
| 초기화 | seed=42의 새 무작위 초기 가중치; 공개 checkpoint 미사용 |
| epoch | 360 |
| batch size | 40 |
| epoch당 처리 이미지 수 | 435; `drop_last=False` |
| 입력·정답 | RGB float32 `[0,1]`, NCHW |
| 학습 crop | 대응 쌍의 동일 위치 **180×180 중앙 crop** |
| crop 외 데이터 변환 | 추가 합성 잡음·flip·회전 없음 |
| optimizer | PyTorch Adam 기본 β=(0.9, 0.999), weight_decay=0 |
| 초기 학습률 | 0.01 |
| 스케줄 | StepLR, 40 epoch마다 ×0.1 |
| CPU 설정 | 기본 intra-op threads=1, inter-op threads=1, DataLoader workers=0; 실행별 실제 값은 `run.json`에 기록 |
| 학습 이미지 순서 | seed=42의 별도 DataLoader generator로 shuffle |
| 검증 | 매 epoch, 추가 합성 잡음 없음, 전체 해상도, batch=1 |
| checkpoint 선택 | 검증 이미지별 PSNR 산술평균 최대; 동점은 앞선 epoch |

Python·NumPy·PyTorch의 시드를 설정하고 `torch.use_deterministic_algorithms(True)`를 사용합니다. 같은 시드만으로 서로 다른 하드웨어·라이브러리 버전의 완전한 수치 일치를 보장하지 않으므로 환경과 초기값 해시를 함께 기록합니다.

공식 `UltraFastLiNETLoss`를 변경하지 않고 사용합니다.

```text
L = 0.975 × SmoothL1(최종 출력, 정답)
  + 0.025 × (1 − MS-SSIM(최종 출력, 정답))
  + 1.0 × 다중 해상도 Sobel gradient loss
```

gradient loss는 decoder 출력 `(H/4, H/2, H)`에 순서대로 `(1.0, 1.0, 0.04)`의 가중치를 사용합니다. 정답을 각 출력 해상도로 bilinear 보간하고 RGB 채널 평균 영상의 가로·세로 Sobel 차이에 Smooth L1을 적용합니다. 손실은 **clamp하지 않은 원시 출력**으로 계산하며, 지표·PNG 저장 때만 출력을 `[0,1]`로 clamp합니다. 기본 5단계 MS-SSIM 때문에 crop의 짧은 변은 160보다 커야 합니다.

B/C 확장 시 모델 구조, 손실, 분할, 시드별 초기 가중치, 이미지 순서, crop, epoch, optimizer, 스케줄과 선택 정책을 공통으로 유지해야 합니다. 현재 A 선택 정책은 `validation_psnr_sigma0`입니다. 잡음 조건을 포함한 다른 선택 정책을 도입하려면 **A에도 같은 정책을 적용하고 그 변경을 명시**해야 합니다.

## 평가 지표 정의

- **PSNR:** RGB 전체 픽셀의 MSE에 대해 `10 log10(1/MSE)`를 계산합니다. 공식 지표 구현은 MSE가 정확히 0이면 100 dB를 반환합니다.
- **SSIM:** `pytorch_msssim.ssim`의 기본 11×11 Gaussian 창, σ=1.5, `data_range=1`, `size_average=True`를 사용합니다. 손실의 **MS-SSIM과 보고용 SSIM은 다른 지표**입니다.
- 검증·시험은 전체 해상도에서 이미지별로 계산하고 그 점수의 산술평균을 보고합니다. 모든 이미지를 합친 MSE에서 계산한 PSNR과는 다릅니다.
- float32 예측을 `[0,1]`로 clamp한 뒤 채점합니다. Y 채널 변환, 테두리 crop, GT에 맞춘 밝기 보정, PNG 8-bit 양자화 후 채점은 적용하지 않습니다.
- 정답은 원래 정상 조도 이미지입니다. 모델 출력에 유리하도록 정답이나 입력을 재정렬·밝기 보정하지 않습니다.

다른 논문이나 구현과 점수를 비교할 때는 데이터 분할뿐 아니라 이 지표 정의도 일치하는지 확인해야 합니다.

## 완료된 예비 측정

### A 모델 1 epoch 파일럿

공개 가중치를 로드하지 않고 원본 입력 435쌍으로 1 epoch 학습한 결과입니다. 검증은 50쌍, 추가 합성 잡음 없음 조건입니다.

| 항목 | 실제 측정 |
|---|---:|
| 평균 학습 loss | 0.229896 |
| 검증 평균 PSNR | 15.2281 dB |
| 검증 평균 SSIM | 0.525913 |
| 학습 시간 | 약 23.31초 |
| 검증 시간 | 약 3.30초 |
| 공식 시험 세트 사용 | 없음 |

이는 학습 루프의 동작 확인이며 최종 성능이 아닙니다. 원자료는 [log.csv](../reports/a_pilot_seed42/log.csv)에서 확인할 수 있습니다. 시간은 이 실행에서 기록한 값이며 여러 번 반복한 벤치마크가 아닙니다.

### 공식 공개 가중치 CPU 추론

아래는 **공개 가중치**를 LOL 원래 학습 이미지 `10.png` 한 장에서 측정한 예비 벤치마크입니다. A의 전체 학습 후 성능과 구분합니다.

| 항목 | 실제 측정 |
|---|---:|
| CPU | Apple M2 Pro |
| 입력 | RGB 600×400, `[1,3,400,600]`, FP32 |
| intra-op / inter-op threads | 1 / 1 |
| warm-up / 측정 반복 | 10 / 50 |
| 전체 3-output forward 중앙값 | 27.27 ms |
| 전체 3-output forward p95 | 28.16 ms |
| 프로세스 peak RSS | 약 319.98 MiB |
| 파라미터 수 / FP32 텐서 크기 | 180 / 720 bytes |
| 공개 checkpoint 파일 크기 | 22,818 bytes |
| 명목 산술량 추정 | 54.28815 M ops |
| THOP 기본 연산 카운트 | 7.76325 M; 전체 FLOPs 아님 |

시간은 파일 읽기, tensor 변환, clamp, 지표 산출, PNG 저장을 제외한 모델 forward입니다. 세 decoder 출력을 모두 계산합니다. 단일 입력·단일 장치 측정이며, 논문의 다른 장치·런타임 수치와 직접 비교하지 않습니다. CPU 스레드 수, 전원 상태, 열 상태와 백그라운드 부하는 후속 비교에서 동일하게 관리해야 합니다.

peak RSS는 Python·PyTorch·라이브러리·활성값을 포함한 프로세스의 최대 상주 메모리입니다. 모델만의 추가 메모리나 가중치 크기와 같지 않습니다. 720 bytes는 FP32 파라미터 텐서만의 크기이고 checkpoint 파일에는 직렬화 부가 정보가 들어갑니다.

산술량 추정은 곱셈과 덧셈을 각각 1회로 세고 pooling의 padding을 포함한 명목 3×3 연산을 사용합니다. sigmoid·ReLU, 복사·인덱싱, nearest 보간·메모리 할당 등은 제외합니다. THOP 기본 카운터는 functional shift, gate, skip 및 branch 연산 일부를 누락하므로 전체 FLOPs로 쓰지 않습니다. 정확한 정의와 반복 측정 원자료는 [metrics.json](../reports/reproduction/metrics.json)에 있습니다.

이 한 장에서 공개 가중치의 PSNR 20.584 dB, SSIM 0.4981도 확인했지만, 공개 가중치가 학습했을 수 있는 원래 학습 이미지이므로 시험 benchmark 표에 넣지 않습니다.

## 저장소와 실행 산출물

```text
.
├── configs/baseline_a.json      # 고정 A 학습 설정
├── manifests/                  # train/validation/test 목록 및 파일 해시
├── scripts/
│   ├── common.py               # 공통 I/O, 데이터셋, 시드, 공식 코드 import
│   ├── prepare_data.py         # 중복 그룹 분할 및 데이터 검증
│   ├── reproduce.py            # 공개 가중치 추론·CPU 초기 측정
│   ├── train_a.py              # 새 초기값에서 A 학습
│   ├── evaluate.py             # 완료된 A의 최종 시험 평가
│   └── summarize_results.py    # 확인된 최종 결과를 README·보고서에 반영
├── tests/                      # 재현·데이터·평가 검증
├── vendor/                     # 고정 upstream 소스와 라이선스
├── weights/official_max.pkl    # 동작 확인용 공식 공개 checkpoint
├── reports/                    # 재현 기록·측정값·최종 평가 기록
├── runs/                       # 로컬 학습 출력; 실행 시 생성
├── provenance.json             # upstream commit·소스·checkpoint 해시
├── data_provenance.json        # 데이터 출처·압축파일 해시
└── requirements.lock.txt       # 확인한 Python 의존성 버전
```

학습 출력 폴더의 주요 파일은 다음과 같습니다.

| 파일 | 내용 |
|---|---|
| `initial.pt` | 새 초기 가중치와 시드 |
| `best.pt` | 검증 평균 PSNR이 가장 높은 checkpoint |
| `last.pt` | 마지막 완료 epoch의 checkpoint |
| `run.json` | 설정·환경·초기값 해시·분할 해시·출처 |
| `log.csv` | 매 epoch loss·PSNR·SSIM·학습률·시간·처리 수 |
| `best_validation_per_image.json` | best checkpoint의 검증 이미지별 점수 |
| `summary.json` | 전체 실행의 완료 상태와 최고 검증 PSNR |

재개용 `last.pt`에는 가중치, optimizer, scheduler, epoch, 설정, 파일럿 표시, PyTorch RNG와 DataLoader generator 상태, 현재까지의 best 가중치·점수가 저장됩니다. `best.pt`에는 선택된 가중치와 epoch·설정·점수 등 평가용 정보가 저장됩니다. `--resume`는 같은 출력 폴더의 `last.pt`를 사용하여 완료된 epoch 다음부터 학습합니다. 설정·분할·학습 소스 해시가 달라진 실행은 이어 붙이지 않습니다.

전체 데이터셋과 가상환경은 GitHub에 포함하지 않습니다. README에는 LOL-v1 테스트 목록의 첫 두 장(`1.png`, `111.png`)에 대한 A 모델의 비교 이미지만 [docs/assets](assets/)에 제공합니다. 데이터는 [공식 출처](https://daooshee.github.io/BMVC2018website/)에서 받으며, 데이터 이미지의 권리는 원 권리자에게 있고 저장소의 코드 라이선스와는 별개입니다.

예시는 기존 최종 평가에 사용한 `reports/a_full_seed42/best.pt`(epoch 20)로 `evaluate.py --save-images`를 실행해 만들었습니다. 전체 테스트 15쌍의 PSNR·SSIM이 기존 평가 기록과 일치하는 것을 확인했습니다. 각 패널은 원본 600×400 픽셀을 그대로 배치했고 밝기 조정·크롭·리사이즈를 하지 않았습니다. 샘플 선택 방식, checkpoint·이미지 해시와 개별 점수는 [examples.json](assets/examples.json)에 기록했습니다.

## 후속 B/C 실험에서 유지할 원칙

1. 동일 시드의 A/B/C는 같은 초기 모델 가중치에서 시작합니다. seed만 같다고 추정하지 말고 텐서 또는 초기값 기록으로 확인합니다.
2. 잡음 RNG와 데이터 순서 RNG를 분리합니다. 증강 코드가 난수를 더 사용해도 이미지 순서와 다른 조건이 바뀌지 않게 합니다.
3. 검증·시험 잡음은 이미지와 조건별로 고정하고 A/B/C에 같은 noisy input을 제공합니다. 학습 중에는 새 잡음을 생성할 수 있습니다.
4. 추가 합성 잡음 없음 조건의 품질 하락과 질감 손실도 함께 보고합니다. 높은 잡음 조건 점수만으로 개선을 주장하지 않습니다.
5. B와 C의 평균 σ는 모두 0.03이지만 평균 분산은 다릅니다. 잡음이 추가된 샘플 기준 `E[σ²]`는 B=0.0009, C≈0.0011667로 C가 약 29.6% 큽니다. 평균 σ 일치만으로 평균 잡음 에너지가 통제되지는 않습니다.
6. CPU 비용은 같은 입력 크기·스레드·측정 범위에서 직접 비교합니다. 구조가 같다는 이유만으로 실행시간이나 peak RSS가 완전히 같다고 쓰지 않습니다.
7. 학습에 없던 Poisson 계열 잡음 또는 LSRW 외부 검증을 추가하면 Gaussian 세기 비교와 구분해 보고합니다. 현재 저장소는 그 실험을 수행하지 않았습니다.

## 라이선스와 인용

이 저장소의 코드 라이선스는 **Apache License 2.0**이며 전문은 [LICENSE](../LICENSE)에 있습니다. 가져온 공식 코드의 원본 라이선스도 [vendor/LICENSE](../vendor/LICENSE)에 보존합니다. 원본 파일의 기여는 공식 UltraFast-LiNET 저자에게 있으며, 이 프로젝트가 추가한 실행용 코드는 `scripts/`, 설정·검증 코드는 `configs/`와 `tests/`에 구분되어 있습니다. 고정 upstream의 파일 해시는 [provenance.json](../provenance.json)으로 확인합니다. 데이터셋의 사용·재배포 조건은 코드 라이선스와 별개입니다.

보고서에서는 [UltraFast-LiNET 논문](https://arxiv.org/abs/2512.02965), [고정한 공식 코드 버전](https://github.com/YuhanChen2024/UltraFast-LiNET/tree/12e8c79566235fecc952d9a148011cf5fbb2c8ba), [LOL 데이터셋 배포 페이지](https://daooshee.github.io/BMVC2018website/)를 출처로 기재하고, 435/50/15 분할과 공개 구현을 유지한 범위를 함께 설명합니다.
