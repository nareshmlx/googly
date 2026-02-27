"""URL safety guards for fulltext enrichment fetches."""

import ipaddress
from urllib.parse import urlparse

_BLOCKED_HOSTNAMES = {
    "localhost",
    "0.0.0.0",
    "127.0.0.1",
    "::1",
}


def _is_private_ip(hostname: str) -> bool:
    """Return True when hostname is an IP address in private/local ranges."""
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def is_safe_public_url(url: str, allowed_domains: set[str] | None = None) -> tuple[bool, str]:
    """Validate URL scheme/host against SSRF guardrails and optional allowlist."""
    parsed = urlparse(str(url or "").strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return False, "invalid_scheme"

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False, "missing_hostname"

    if hostname in _BLOCKED_HOSTNAMES or hostname.endswith(".local") or hostname.endswith(".internal"):
        return False, "blocked_hostname"

    if _is_private_ip(hostname):
        return False, "private_ip"

    if allowed_domains and not any(
        hostname == domain or hostname.endswith(f".{domain}") for domain in allowed_domains
    ):
        return False, "domain_not_allowed"

    return True, "ok"
