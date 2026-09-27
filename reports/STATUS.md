> **최신 완료 상태:** 360 epoch 전체 학습 및 공식 시험 평가 완료. [최종 보고서](FINAL_RESULTS.md)를 참고하세요. 아래는 초기 재현 당시의 기록입니다.

# 기준 모델 재현 진행 기록

## 현재 상태: 360 epoch 학습 진행 중

갱신일: 2026-09-26 (Asia/Seoul)

**A 모델의 360 epoch 전체 학습을 실행 중입니다.** 이 문서 갱신 시 전체 실행의 79 epoch까지 학습 로그가 기록되었습니다. 최신 완료 epoch는 [전체 학습 log.csv](a_full_seed42/log.csv), 고정 설정과 환경은 [run.json](a_full_seed42/run.json)을 확인합니다. 전체 학습 완료 및 공식 시험 15쌍의 최종 성능은 아직 대기 중이며, 아래 1 epoch 파일럿 수치는 최종 기준선이 아닙니다.

전체 실행은 공개 가중치나 파일럿 checkpoint를 이어 쓰지 않고 seed=42의 새 초기 가중치에서 시작했습니다. 435쌍 학습·50쌍 검증, 중앙 crop 180×180, 공식 구조·손실, Adam 및 40 epoch마다 감쇠하는 학습률 일정을 유지합니다. 학습 완료 후 검증 평균 PSNR로 선택한 `best.pt`만 공식 시험 15쌍에서 평가합니다. 시험 점수로 checkpoint를 선택하지 않습니다.

실행 방법과 최종 결과 표는 [README](../README.md)에 정리했습니다. 실행 소스는 [train_a.py](../scripts/train_a.py)와 [evaluate.py](../scripts/evaluate.py), 설정은 [baseline_a.json](../configs/baseline_a.json), 출처·원본 파일 해시는 [provenance.json](../provenance.json)에 있습니다. 재개용 checkpoint는 epoch 경계에서 저장하고 `--resume`으로 복원합니다. 최종 시험 결과를 확인한 뒤 B/C 정책을 시험 결과에 맞춰 조정하지 않으며, 설정 결정에는 검증 세트를 사용합니다.

## 초기 재현 기록: 2026-09-25

아래 내용은 **최초 동작 확인 당시의 역사 기록**입니다. 당시 공식 UltraFast-LiNET-Max를 CPU에서 실행해 보정 이미지를 로컬에 저장하고 파라미터 수를 확인했습니다. 데이터 분할과 새 초기 가중치에서의 A 모델 1 epoch 학습도 완료했습니다. 이 시점에는 전체 학습과 A/B/C 비교를 수행하지 않았습니다.

## 실제 실행 결과

| 항목 | 확인 결과 |
|---|---|
| 공식 코드 버전 | `12e8c79566235fecc952d9a148011cf5fbb2c8ba` |
| 모델 | UltraFast-LiNET-Max, 공식 checkpoint 호환 구조 |
| 전체 / 학습 가능 파라미터 | **180 / 180** |
| FP32 파라미터 텐서 크기 | **720 bytes** |
| 공개 checkpoint 파일 크기 | 22,818 bytes; 모델 텐서 크기와 다름 |
| 공개 checkpoint 로드 | strict=True, 누락·추가 키 없음 |
| 입력 | LOL-v1 원래 학습 세트 `10.png`, RGB, 600×400 |
| 입력 tensor | `[1,3,400,600]`, float32, 실제 범위 [0,0.2] |
| 출력 tensor | `[1,3,100,150]`, `[1,3,200,300]`, `[1,3,400,600]` |
| 최종 원시 출력 범위 | 약 [0.03473,1.28804]; 저장·지표 계산 시 [0,1]로 clamp |
| CPU | Apple M2 Pro, thread=1, batch=1, FP32 |
| CPU forward 중앙값 / p95 | **27.27 / 28.16 ms** |
| 측정 방식 | warm-up 10회, 50회 반복, 세 출력 전체 forward, I/O·후처리 제외 |
| 프로세스 peak RSS | 약 **319.98 MiB**; Python·PyTorch·라이브러리 포함 |
| 산술량 추정 | 54.28815 M ops; 곱셈·덧셈 각 1, pooling은 padding 포함 명목 3×3; 활성화·복사 등 제외 |
| THOP 기본값 | 7.76325 M; shift/gate/skip/branch 연산 누락 때문에 전체 FLOPs가 아님 |

정확한 설정·시간 50개 원자료·연산별 내역은 [metrics.json](reproduction/metrics.json)에 있습니다. CPU 결과는 이 입력과 현재 장치에 대한 초기 측정이며 여러 장치·다중 시드의 배포 성능 평가를 대신하지 않습니다. 파라미터 수와 프로세스 메모리는 다른 지표입니다.

입력·공개 가중치 출력·정답 비교 이미지는 로컬 실행으로 생성했으며 GitHub에는 포함하지 않습니다. 동일 비교는 [실행 가이드의 공개 checkpoint 추론 명령](../docs/EXPERIMENT_GUIDE.md#2-공개-checkpoint의-cpu-추론-확인)으로 생성할 수 있습니다.

이 비교는 **추가 합성 잡음 없음** 조건입니다. 공개 가중치 출력에서는 밝기 회복과 함께 색 잡음이 드러납니다. 이 한 장만으로 잡음 취약성 전체를 결론내리지는 않습니다. 이 이미지에 대한 출력 PSNR 20.584 dB, SSIM 0.4981은 이미 학습에 사용되었을 수 있는 샘플의 동작 확인값입니다. 논문의 LOL 시험 점수 19.81 dB / 0.73과 비교하는 실험값이 아닙니다.

## 데이터 분할 완료

LOL 공식 배포본 485/15쌍을 확보했습니다. 학습·검증·시험은 **435/50/15**로 고정했습니다. 쌍별 파일명, RGB 모드, 해상도 및 이미지 무결성을 학습 485쌍에서 확인했습니다. 이 초기 재현 시점의 공식 시험 세트는 이름과 SHA-256만 확인했습니다.

정상 조도 파일이 바이트 단위로 같은 **66개 그룹, 총 152쌍**을 발견했습니다. 단순 무작위 분할이 이 중복 정답을 양쪽에 배치하는 것을 확인해, 실제 학습 전에 중복 그룹 단위 분할로 교체했습니다. train/validation/test 사이에 같은 이름 또는 같은 low/high 파일 해시가 겹치지 않습니다. 장면 유사성 자체를 전수 조사한 것은 아닙니다.

목록과 출처는 [manifests](../manifests/)와 [data_provenance.json](../data_provenance.json)에 기록했습니다. 초기 파일명 기반 분할은 폐기했으며, 그 분할로 모델을 업데이트한 적이 없습니다.

## A 모델 학습 동작 확인

공개 가중치를 사용하지 않고 seed=42로 초기화했습니다. 원본 입력 435쌍, 중앙 crop 180×180, batch=40, Adam lr=0.01, 공식 손실을 사용했습니다. 1 epoch 동안 435쌍 전체를 처리했고, 모든 파라미터에 유한한 gradient가 전달되는지 확인했습니다.

| 지표 | 1 epoch 파일럿 결과 |
|---|---:|
| 평균 학습 loss | 0.229896 |
| 검증 50쌍 평균 PSNR | **15.2281 dB** |
| 검증 50쌍 평균 SSIM | **0.525913** |
| 학습 / 검증 시간 | 약 23.31 / 3.30초 |
| 최종 시험 세트 사용 | 없음 |

학습 시간은 파일럿 실행 기록이며 별도 반복 측정값이 아닙니다. 결과는 [log.csv](a_pilot_seed42/log.csv), [run.json](a_pilot_seed42/run.json), `initial.pt`, `best.pt`, `last.pt`에 있습니다. 모든 체크포인트에 `pilot=true`를 기록했습니다. **360 epoch 학습을 마친 A 기준선으로 사용하지 않습니다.**

## 재현에서 발견한 차이와 처리

- 공개 코드의 디코더 경로는 논문 식의 순차 연결과 다릅니다. `vendor/model.py`의 설명대로 공개 가중치 호환 구조를 유지했습니다. 공식 dia 기본값 1…5도 그대로 사용했습니다.
- 공식 학습 예시는 시험 데이터로 최적 모델을 고릅니다. 새 `train_a.py`는 train/validation manifest만 사용하고 `split=test`를 거부합니다.
- 공개 checkpoint에는 `epoch=120`만 기록되어 있습니다. 논문의 360 epoch 학습 이력 전체를 재현한 것으로 간주하지 않습니다.
- Python 3.12에서 오래된 THOP가 `distutils`를 찾지 못해 `setuptools==80.9.0`을 고정했습니다. 설치 버전 전체는 `requirements.lock.txt`에 있습니다.
- 논문의 FLOPs 표·본문 및 실제 계산 경로에 차이가 있어, 이번 보고서에는 연산 정의를 명시한 별도 추정값을 사용했습니다.

출처: [공식 저장소](https://github.com/YuhanChen2024/UltraFast-LiNET/tree/12e8c79566235fecc952d9a148011cf5fbb2c8ba), [논문 §3](https://arxiv.org/html/2512.02965v2#S3), [LOL 배포 페이지](https://daooshee.github.io/BMVC2018website/).

## 초기 재현 당시의 다음 실행 계획

이 목록은 2026-09-25에 정한 순서입니다. 현재 1번 전체 학습은 진행 중이며, 최신 상태는 문서 상단을 확인합니다.

1. [README](../README.md)의 명령으로 A를 **처음부터 360 epoch** 학습합니다. 파일럿 checkpoint를 이어 학습하는 대신 동일 seed의 새 실행으로 시작합니다.
2. 검증 세트에서 전처리·밝기·PSNR·SSIM 및 잡음 세기별 출력을 확인합니다.
3. B/C 구현 전에 공통 checkpoint 선택 규칙과 잡음 RNG·평가 규칙을 고정합니다. 최종 시험 15쌍은 계속 최종 평가에만 사용합니다.

초기 재현 시점의 자동 검증 **10개가 모두 통과**했습니다([test-results.txt](test-results.txt)). THOP의 distutils 사용에 관한 deprecation warning 4개가 있으며 실행 실패는 없습니다. 공식 소스 해시, checkpoint 호환성, 홀수 해상도, shift 연산, 전체 gradient, 초기화 재현성, 지표, 대응 crop, 잘못된 쌍 거부, 중복 분리, 연산량 집계가 검증 대상입니다.

공식 `vendor/test.py`도 같은 훈련 이미지로 직접 실행했습니다. 생성 PNG와 새 추론 코드의 PNG가 **모든 픽셀에서 동일**했습니다([official_cli_parity.json](official_cli_parity.json), 최대 uint8 픽셀 오차 0).

공식 코드의 출처와 라이선스는 [고정 upstream](https://github.com/YuhanChen2024/UltraFast-LiNET/tree/12e8c79566235fecc952d9a148011cf5fbb2c8ba), [원본 라이선스](../vendor/LICENSE), [저장소 라이선스](../LICENSE)에서 확인할 수 있습니다. 이 초기 공개 모델 진단의 비교 PNG는 로컬에 보관합니다. 이후 완료한 A 모델의 테스트 예시 두 장은 [README](../README.md#보정-예시)에 추가했습니다.
