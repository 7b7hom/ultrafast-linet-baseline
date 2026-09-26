"""Update human-readable results only from a sealed full run and final evaluation."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from common import ROOT, sha256


def summarize(run_dir, evaluation_dir, project=ROOT):
    project, run_dir, evaluation_dir = map(Path, (project, run_dir, evaluation_dir))
    summary = json.loads((run_dir / "summary.json").read_text())
    metrics = json.loads((evaluation_dir / "metrics.json").read_text())
    if (summary.get("completed_epochs") != 360 or summary.get("pilot") is not False
            or summary.get("full_baseline_complete") is not True
            or metrics.get("count") != 15 or metrics.get("official_test_used") is not True
            or metrics.get("test_used_for_checkpoint_selection") is not False
            or metrics.get("additional_synthetic_noise_sigma") != 0):
        raise ValueError("Only a completed 360-epoch run and final 15-pair sigma=0 evaluation can be published")
    verification = metrics["verification"]
    if verification["checkpoint_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("Training/evaluation checkpoint hashes disagree")
    for kind in ("best", "last"):
        if sha256(run_dir / f"{kind}.pt") != summary["checkpoint_sha256"][kind]:
            raise ValueError("Checkpoint changed after final evaluation")
    if sha256(run_dir / "log.csv") != verification["training_log_sha256"]:
        raise ValueError("Training log changed after final evaluation")
    with (run_dir / "log.csv").open(newline="") as handle:
        log = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in log] != list(range(1, 361)):
        raise ValueError("Incomplete epoch history")
    chosen = max(log, key=lambda row: float(row["val_psnr"]))
    epoch = int(chosen["epoch"])
    if epoch != summary["best_epoch"] or epoch != verification["selected_epoch"]:
        raise ValueError("Validation selection does not match final evaluation")
    train_seconds = sum(float(row["train_seconds"]) for row in log)
    val_seconds = sum(float(row["val_seconds"]) for row in log)
    score = metrics["aggregate"]
    digest = summary["checkpoint_sha256"]["best"]
    table = f"""| 항목 | 결과 |
|---|---|
| 완료 epoch / 목표 epoch | **360 / 360** |
| 검증으로 선택한 epoch | **{epoch}** |
| 선택 시 검증 50쌍 평균 PSNR / SSIM | **{float(chosen['val_psnr']):.4f} dB / {float(chosen['val_ssim']):.6f}** |
| 공식 시험 15쌍 평균 PSNR / SSIM, 추가 합성 잡음 없음 | **{score['psnr']:.4f} dB / {score['ssim']:.6f}** |
| 선택된 checkpoint SHA-256 | `{digest}` |
| 로그에 기록된 학습·검증 시간 합 | {(train_seconds + val_seconds) / 3600:.3f}시간; 절전·대기 포함 가능 |
"""
    readme_path = project / "README.md"
    readme = readme_path.read_text()
    start, end = "<!-- FINAL_RESULTS_START -->", "<!-- FINAL_RESULTS_END -->"
    if readme.count(start) != 1 or readme.count(end) != 1:
        raise ValueError("README final-results markers missing or ambiguous")
    readme = re.sub(re.escape(start) + r".*?" + re.escape(end),
                    start + "\n\n" + table + "\n" + end, readme, flags=re.DOTALL)
    lines = []
    for line in readme.splitlines():
        if line.startswith("**2026-09-26 현재 A 모델의"):
            line = "**A 모델의 360 epoch 전체 학습과 공식 시험 15쌍의 최종 평가를 완료했습니다.** 검증 세트만으로 선택한 checkpoint를 사용했습니다. [최종 보고서](reports/FINAL_RESULTS.md)와 [이미지별 시험 결과](reports/a_final_seed42/per_image.csv)를 함께 확인하세요."
        elif line.startswith("| A 모델 360 epoch 전체 학습 |"):
            line = "| A 모델 360 epoch 전체 학습 | **완료** | 435쌍 × 360 epoch, seed=42 |"
        elif line.startswith("| 검증으로 선택한 A의 공식 시험 15쌍 평가 |"):
            line = "| 검증으로 선택한 A의 공식 시험 15쌍 평가 | **완료** | 추가 합성 잡음 없음, checkpoint 선택에 시험 데이터 미사용 |"
        elif line.startswith("완료 여부와 수치는 실행 산출물 검증 후 갱신합니다."):
            line = "아래 값은 완료 기록, 가중치 해시, 검증 선택 및 최종 평가 파일을 대조한 결과입니다."
        lines.append(line)
    readme_path.write_text("\n".join(lines) + "\n")
    created = datetime.now(timezone.utc).isoformat()
    report = f"""# A 기준 모델 전체 학습 및 최종 평가

생성 시각: {created}

전체 학습 360 epoch를 완료했습니다. 최종 시험은 학습이 끝난 뒤 검증 PSNR이 가장 높은 **epoch {epoch}**의 checkpoint로 수행했습니다. 시험 점수로 모델을 다시 선택하거나 설정을 조정하지 않았습니다.

{table}

## 실행 조건

- UltraFast-LiNET-Max 공식 checkpoint 호환 구조, 180개 파라미터, 공식 손실 유지.
- LOL-v1 train 435 / validation 50 / official test 15, 중복 정답 그룹이 분할을 넘지 않도록 고정.
- seed=42의 새 초기값, 추가 잡음 없음, 대응 중앙 crop 180×180, batch=40.
- Adam lr=0.01, 40 epoch마다 ×0.1, 총 360 epoch. 매 epoch 435쌍, 총 156,600쌍 처리.
- CPU float32, threads=1. optimizer·scheduler·초기값·실행 코드·분할 기록은 [run.json](a_full_seed42/run.json)에 있습니다.
- 검증의 이미지별 PSNR 평균 최대를 선택하며 동점은 앞선 epoch를 유지합니다.
- 시험은 RGB 전체 해상도, 예측 clamp [0,1], 이미지별 PSNR/SSIM의 산술평균입니다. Y 채널 전환·GT 밝기 보정·테두리 제거를 하지 않았습니다.

## 결과 파일

- [360 epoch 로그](a_full_seed42/log.csv)
- [학습 완료 요약 및 checkpoint 해시](a_full_seed42/summary.json)
- [최종 시험 지표 및 검증 기록](a_final_seed42/metrics.json)
- [시험 이미지별 CSV](a_final_seed42/per_image.csv)
- [선택한 best.pt](a_full_seed42/best.pt)

## 해석과 한계

이 결과는 단일 시드의 A 기준선입니다. B/C 잡음 증강의 효과는 아직 측정하지 않았습니다. 공개 코드의 디코더 연결과 논문 수식에 차이가 있고, 원래 학습 485쌍에서 검증 50쌍을 분리했으므로 논문 19.81 dB와 같은 조건의 직접 재현으로 주장하지 않습니다. 논문 수치와 차이가 있어도 시험 결과를 이용해 이번 선택 규칙을 변경하지 않습니다.

기록된 학습 구간 시간 합은 {train_seconds:.2f}초, 검증 구간 합은 {val_seconds:.2f}초입니다. 장시간 실행 중 절전·일시정지·백그라운드 부하가 포함될 수 있어 순수 연산 성능 수치로 해석하지 않습니다. CPU 추론 지연의 별도 예비 측정은 [reproduction/metrics.json](reproduction/metrics.json)에 있습니다.

공식 시험 결과는 이제 확인된 상태입니다. 후속 B/C의 잡음 범위·학습률·선택 기준은 검증 세트에서 결정하고, 시험 결과를 보고 조정했다면 그 사실을 별도로 보고해야 합니다.
"""
    (project / "reports/FINAL_RESULTS.md").write_text(report)
    status_path = project / "reports/STATUS.md"
    status = status_path.read_text()
    notice = "> **최신 완료 상태:** 360 epoch 전체 학습 및 공식 시험 평가 완료. [최종 보고서](FINAL_RESULTS.md)를 참고하세요. 아래는 초기 재현 당시의 기록입니다.\n\n"
    if not status.startswith(notice):
        status_path.write_text(notice + status)
    return {"selected_epoch": epoch, "test_psnr": score["psnr"], "test_ssim": score["ssim"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/a_full_seed42")
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "reports/a_final_seed42")
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir, args.evaluation_dir), indent=2))
