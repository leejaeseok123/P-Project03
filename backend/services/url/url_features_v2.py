
import ipaddress
from urllib.parse import urlsplit
import pandas as pd
import tldextract

extractor = tldextract.TLDExtract(
    suffix_list_urls=()
)

FEATURES_V2 = [
    "URLLength",
    "DomainLength",
    "PathLength",
    "QueryLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "NoOfDigitsInURL",
    "DigitRatioInURL",
    "NoOfHyphens",
    "NoOfDots",
    "NoOfEquals",
    "NoOfQMark",
    "NoOfAmpersand",
    "IsHTTPS",
    "SchemeKnown",
]


def extract_features_v2(url):
    url = str(url).strip()

    # 실제 프로토콜 유무 확인
    scheme_known = int(
        url.lower().startswith(
            ("http://", "https://")
        )
    )

    # 파싱을 위한 임시 프로토콜
    parse_url = (
        url if scheme_known
        else "http://" + url
    )

    parsed = urlsplit(parse_url)
    host = (parsed.hostname or "").lower()

    if not host:
        raise ValueError("호스트가 없습니다.")

    ext = extractor(host)

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    subdomains = (
        ext.subdomain.split(".")
        if ext.subdomain else []
    )

    digits = sum(c.isascii() and c.isdigit() for c in url)
    letters = sum(c.isascii() and c.isalpha() for c in url)

    # 미상은 -1로 인코딩
    is_https = (
        int(parsed.scheme.lower() == "https")
        if scheme_known else -1
    )

    return {
        "URLLength": len(url),
        "DomainLength": len(host),
        "PathLength": len(parsed.path),
        "QueryLength": len(parsed.query),
        "IsDomainIP": is_ip,
        "TLDLength": len(ext.suffix),
        "NoOfSubDomain": len(subdomains),
        "NoOfLettersInURL": letters,
        "NoOfDigitsInURL": digits,
        "DigitRatioInURL": digits / max(len(url), 1),
        "NoOfHyphens": url.count("-"),
        "NoOfDots": url.count("."),
        "NoOfEquals": url.count("="),
        "NoOfQMark": url.count("?"),
        "NoOfAmpersand": url.count("&"),
        "IsHTTPS": is_https,
        "SchemeKnown": scheme_known,
    }


def transform_urls_v2(urls):
    rows = [extract_features_v2(url) for url in urls]
    return pd.DataFrame(rows, columns=FEATURES_V2)


if __name__ == "__main__":
    test_urls = [
        "https://example.com/login",
        "http://example.com/login",
        "example.com/login",
        "https://bank-security.example/verify?id=12345"
    ]

    for url in test_urls:
        print(f"\nURL: {url}")
        print(extract_features_v2(url))