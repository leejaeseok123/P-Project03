from urllib.parse import urlsplit

from .predict_url_v2 import predict_url
from .url_features_v2 import extract_features_v2


def _risk_level(score: float, threshold: float) -> str:
    if score >= max(0.8, threshold):
        return "high"
    if score >= threshold:
        return "medium"
    return "low"


def _suspicious_features(url: str) -> list[str]:
    features = extract_features_v2(url)
    reasons = []

    if features["IsDomainIP"]:
        reasons.append("도메인 대신 IP 주소를 사용함")
    if features["SchemeKnown"] == 0:
        reasons.append("URL 프로토콜이 명시되지 않음")
    elif features["IsHTTPS"] == 0:
        reasons.append("HTTPS를 사용하지 않음")
    if features["NoOfSubDomain"] >= 3:
        reasons.append("하위 도메인이 과도하게 많음")
    if features["URLLength"] >= 100:
        reasons.append("URL 길이가 비정상적으로 김")
    if features["DigitRatioInURL"] >= 0.3:
        reasons.append("URL에 숫자가 과도하게 포함됨")
    if features["NoOfHyphens"] >= 3:
        reasons.append("URL에 하이픈이 과도하게 포함됨")

    return reasons


def analyze_url(request_id: str, url: str) -> dict:
    prediction = predict_url(url)
    score = float(prediction["phishing_score"])
    threshold = float(prediction["threshold"])
    risk_level = _risk_level(score, threshold)
    suspicious_features = _suspicious_features(url)

    if risk_level != "low" and not suspicious_features:
        suspicious_features.append("학습된 피싱 URL 패턴과 유사함")

    normalized_url = url if "://" in url else f"http://{url}"
    domain = urlsplit(normalized_url).hostname or ""
    summaries = {
        "high": "피싱 가능성이 높은 URL입니다.",
        "medium": "주의가 필요한 URL입니다.",
        "low": "피싱 가능성이 낮은 URL입니다.",
    }

    return {
        "id": request_id,
        "risk_level": risk_level,
        "score": score,
        "summary": summaries[risk_level],
        "reasons": suspicious_features if risk_level != "low" else [],
        "details": {
            "domain": domain,
            "blacklisted": None,
            "suspicious_features": suspicious_features,
        },
    }
