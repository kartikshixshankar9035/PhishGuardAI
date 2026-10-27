import pytest

from src.url_analyzer import analyze_url


def test_normal_https_url():
    result = analyze_url("https://www.example.com")

    assert result["hostname"] == "www.example.com"
    assert result["risk_score"] == 0
    assert result["risk_level"] == "Low"


def test_http_ip_address():
    result = analyze_url("http://192.0.2.1/login")

    assert result["is_ip_address"] is True
    assert result["risk_score"] == 40
    assert result["risk_level"] == "Moderate"


def test_multiple_subdomains():
    result = analyze_url(
        "https://login.account.verify.example.com"
    )

    assert result["risk_score"] == 15
    assert any(
        "several subdomain levels" in message
        for _, message in result["findings"]
    )


def test_shortening_service():
    result = analyze_url("https://bit.ly/example")

    assert result["hostname"] == "bit.ly"
    assert result["risk_score"] == 15


def test_empty_url():
    with pytest.raises(ValueError, match="Please enter a URL"):
        analyze_url("")


def test_unsupported_scheme():
    with pytest.raises(ValueError, match="Only HTTP and HTTPS"):
        analyze_url("ftp://example.com")


def test_url_without_scheme():
    result = analyze_url("www.example.com")

    assert result["normalized_url"] == "https://www.example.com"
    assert result["risk_score"] == 0


def test_risk_score_is_capped_at_100():
    result = analyze_url(
        "http://xn--example-9za.com/a%20b/" + "x" * 100
    )

    assert 0 <= result["risk_score"] <= 100
