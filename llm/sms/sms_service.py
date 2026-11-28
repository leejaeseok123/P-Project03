import re
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent / "model" / "sms_baseline.pkl"
_bundle = joblib.load(MODEL_PATH)
vec = _bundle["vec"]
model = _bundle["model"]

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")

RULES = {
    "가족·지인 사칭": ["엄마", "아빠", "폰 고장", "폰이 고장", "액정", "임시폰", "번호 바뀌", "친구 폰"],
    "기관 사칭": ["검찰", "수사관", "금융감독원", "경찰", "국세청", "법원"],
    "송금·결제 요구": ["송금", "입금", "계좌", "이체", "기프트카드", "문화상품권", "상품권"],
    "개인정보 요구": ["신분증", "카드 사진", "비밀번호", "인증번호", "주민등록"],
    "앱 설치 유도": ["앱 설치", "앱 하나", "설치해", "원격"],
    "택배·배송 사칭": ["택배", "배송", "주소 수정", "반송"],
    "고수익 미끼": ["고수익", "일당", "리딩방", "원금 보장", "부업"],
}


def get_risk_level(score):
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


SUMMARY = {
    "high": "스미싱 의심",
    "medium": "주의가 필요한 문자",
    "low": "정상 문자로 보임",
    "unknown": "판정 보류",
}


def analyze_sms(id, text):
    if not text or not text.strip():
        return {
            "id": id,
            "risk_level": "unknown",
            "score": None,
            "summary": SUMMARY["unknown"],
            "reasons": ["분석할 문자 내용이 없습니다"],
            "details": {"urls": [], "suspicious_texts": []},
        }

    score = float(model.predict_proba(vec.transform([text]))[0][1])
    risk_level = get_risk_level(score)

    reasons, suspicious_texts = [], []
    if risk_level != "low":
        for reason, keywords in RULES.items():
            found = [k for k in keywords if k in text]
            if found:
                reasons.append(reason)
                suspicious_texts.extend(found)
        if not reasons:
            reasons.append("기존 스미싱 문자와 표현이 유사함")

    return {
        "id": id,
        "risk_level": risk_level,
        "score": round(score, 3),
        "summary": SUMMARY[risk_level],
        "reasons": reasons,
        "details": {
            "urls": URL_PATTERN.findall(text),
            "suspicious_texts": suspicious_texts,
        },
    }


if __name__ == "__main__":
    tests = [
        "엄마 나 폰 액정 깨져서 임시폰이야 문화상품권 10만원짜리 사서 사진 보내줘",
        "[Web발신] [KB국민] 11/26 14:32 입금 350,000원 잔액 1,284,500원",
        "오늘 저녁 뭐 먹을래? ㅋㅋ",
    ]
    for i, t in enumerate(tests):
        print(analyze_sms(f"test-{i}", t))