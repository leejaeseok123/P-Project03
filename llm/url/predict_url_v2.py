
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

if __package__:
    from .url_features_v2 import extract_features_v2
else:
    from url_features_v2 import extract_features_v2


# ==================================================
# 1. 모델 및 설정 파일 경로
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "v2" / "V2_A_All.json"
CONFIG_PATH = BASE_DIR / "models" / "v2" / "V2_A_All_config.json"


# ==================================================
# 2. 학습된 XGBoost 모델 불러오기
# ==================================================
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"모델 파일이 없습니다: {MODEL_PATH}"
    )

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"설정 파일이 없습니다: {CONFIG_PATH}"
    )

model = XGBClassifier()
model.load_model(str(MODEL_PATH))

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

FEATURES = config["features"]
THRESHOLD = float(config["threshold"])


# ==================================================
# 3. URL 피싱 탐지 함수
# ==================================================
def predict_url(url: str) -> dict:

    # URL 입력 검증
    if not isinstance(url, str):
        raise ValueError("URL은 문자열이어야 합니다.")

    url = url.strip()

    if not url:
        raise ValueError("URL을 입력해주세요.")

    if len(url) > 8192:
        raise ValueError("URL 길이가 너무 깁니다.")

    # URL 특징 추출
    try:
        features = extract_features_v2(url)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"URL 특징 추출 실패: {e}"
        ) from e

    # 학습에 사용한 특징 순서 유지
    X = pd.DataFrame(
        [features],
        columns=FEATURES
    )

    # 피싱(label=1) 예측 점수
    phishing_score = float(
        model.predict_proba(X)[0, 1]
    )

    # 피싱 여부 판단
    is_suspicious = phishing_score >= THRESHOLD

    # 결과 반환
    return {
        "url": url,
        "phishing_score": round(phishing_score, 4),
        "threshold": round(THRESHOLD, 4),
        "is_suspicious": bool(is_suspicious),
        "risk_level": (
            "SUSPICIOUS" if is_suspicious else "LOW"
        )
    }


# ==================================================
# 4. 터미널 테스트
# ==================================================
if __name__ == "__main__":

    print("=" * 50)
    print("URL Phishing Detection - XGBoost V2")
    print(f"판단 임계값: {THRESHOLD:.2f}")
    print("종료하려면 q를 입력하세요.")
    print("=" * 50)

    while True:

        try:
            url = input("\nURL 입력 (종료: q): ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n프로그램 종료")
            break

        if url.lower() == "q":
            print("프로그램 종료")
            break

        # 빈 문자열 처리
        if not url:
            print("URL을 입력해주세요.")
            continue

        try:
            result = predict_url(url)

            print("\n===== 분석 결과 =====")
            print(f"URL: {result['url']}")
            print(
                f"피싱 점수: "
                f"{result['phishing_score']:.4f}"
            )
            print(
                f"판단 임계값: "
                f"{result['threshold']:.2f}"
            )
            print(
                f"피싱 의심 여부: "
                f"{result['is_suspicious']}"
            )
            print(
                f"위험 수준: "
                f"{result['risk_level']}"
            )

        except ValueError as e:
            print(f"입력 오류: {e}")

        except Exception as e:
            print(f"예측 오류: {e}")
