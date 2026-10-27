from urllib.parse import urlparse
import ipaddress
import re


SHORTENING_SERVICES = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "is.gd",
    "cutt.ly",
    "rb.gy",
    "shorturl.at",
}


def analyze_url(raw_url):
    raw_url = raw_url.strip()

    if not raw_url:
        raise ValueError("Please enter a URL.")

    candidate = raw_url if "://" in raw_url else "https://" + raw_url

    try:
        parsed = urlparse(candidate)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise ValueError("The URL format is invalid.")

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    if not hostname:
        raise ValueError("The URL does not contain a valid hostname.")

    hostname = hostname.lower().rstrip(".")
    findings = []
    risk_score = 0

    try:
        ipaddress.ip_address(hostname)
        is_ip = True
    except ValueError:
        is_ip = False

    if len(raw_url) > 75:
        findings.append(("Caution", "The URL is unusually long."))
        risk_score += 10

    if parsed.scheme == "https":
        findings.append(
            ("Info", "The URL uses HTTPS. This does not guarantee the site is safe.")
        )
    else:
        findings.append(("Caution", "The URL uses HTTP rather than HTTPS."))
        risk_score += 15

    if is_ip:
        findings.append(
            ("Caution", "The hostname is an IP address rather than a domain name.")
        )
        risk_score += 25
    elif hostname in SHORTENING_SERVICES:
        findings.append(
            ("Caution", "The hostname belongs to a known URL-shortening service.")
        )
        risk_score += 15

    if "@" in parsed.netloc:
        findings.append(
            ("Caution", "The URL contains an @ symbol in its authority section.")
        )
        risk_score += 20

    if not is_ip and (
        hostname.startswith("xn--") or ".xn--" in hostname
    ):
        findings.append(
            ("Caution", "The domain contains internationalized-domain encoding.")
        )
        risk_score += 15

    if not is_ip and len(hostname.split(".")) >= 4:
        findings.append(
            ("Caution", "The hostname contains several subdomain levels.")
        )
        risk_score += 15

    if not is_ip and "-" in hostname:
        findings.append(("Info", "The hostname contains a hyphen."))
        risk_score += 5

    if re.search(r"%[0-9a-fA-F]{2}", raw_url):
        findings.append(
            ("Info", "The URL contains percent-encoded characters.")
        )
        risk_score += 5

    risk_score = min(risk_score, 100)

    if risk_score < 20:
        risk_level = "Low"
    elif risk_score < 50:
        risk_level = "Moderate"
    else:
        risk_level = "Elevated"

    if not findings:
        findings.append(
            ("Info", "No checks in this basic analyzer were triggered.")
        )

    return {
        "input_url": raw_url,
        "normalized_url": candidate,
        "hostname": hostname,
        "is_ip_address": is_ip,
        "findings": findings,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


if __name__ == "__main__":
    examples = [
        "https://www.example.com",
        "http://192.0.2.1/login",
        "https://login.account.verify.example.com",
        "https://bit.ly/example",
    ]

    for example in examples:
        print("\nURL:", example)

        try:
            result = analyze_url(example)
            print("Hostname:", result["hostname"])
            print(
                "Heuristic risk score:",
                f"{result['risk_score']}/100",
                f"({result['risk_level']})",
            )

            for level, message in result["findings"]:
                print(f"- [{level}] {message}")

        except ValueError as error:
            print("Invalid URL:", error)
