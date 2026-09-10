from __future__ import annotations

import ipaddress
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import requests

ALLOWED_FORMATS = {
    "biomarkers": {"dicom", "pdf", "jpg", "jpeg", "png"},
    "ecg": {"dicom", "pdf", "jpg", "jpeg", "png"},
    "echo": {"dicom", "mp4", "avi"},
}


class DownloadError(ValueError):
    pass


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise DownloadError("Only HTTPS URLs with a hostname are accepted.")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address and (address.is_private or address.is_loopback or address.is_link_local):
        raise DownloadError("Private and loopback download targets are not allowed.")


def _format_from_bytes(prefix: bytes, content_type: str, suffix: str) -> str | None:
    if len(prefix) >= 132 and prefix[128:132] == b"DICM":
        return "dicom"
    if prefix.startswith(b"%PDF-"):
        return "pdf"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
        return "mp4"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"AVI ":
        return "avi"
    content_type = content_type.lower().split(";", 1)[0].strip()
    content_map = {
        "application/pdf": "pdf",
        "image/jpeg": "jpg",
        "image/png": "png",
        "video/mp4": "mp4",
        "video/x-msvideo": "avi",
        "application/dicom": "dicom",
        "application/dicom+json": "dicom",
    }
    return content_map.get(content_type) or suffix.lstrip(".").lower() or None


@contextmanager
def download_to_temp(
    url: str,
    modality: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = 50 * 1024 * 1024,
):
    """Stream a signed/public Cloudinary asset into a request-local temp file."""
    _validate_url(url)
    if modality not in ALLOWED_FORMATS:
        raise DownloadError(f"Unknown modality: {modality}")

    response = requests.get(url, stream=True, timeout=timeout, allow_redirects=False)
    response.raise_for_status()
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        raise DownloadError("Downloaded file exceeds the configured size limit.")

    with tempfile.NamedTemporaryFile(prefix=f"cardiofusion-{modality}-", delete=False) as handle:
        path = Path(handle.name)
        total = 0
        first_chunk = b""
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            if not first_chunk:
                first_chunk = chunk[:512]
            total += len(chunk)
            if total > max_bytes:
                path.unlink(missing_ok=True)
                raise DownloadError("Downloaded file exceeds the configured size limit.")
            handle.write(chunk)

    detected = _format_from_bytes(first_chunk, response.headers.get("content-type", ""), path.suffix)
    if detected not in ALLOWED_FORMATS[modality]:
        try:
            import pydicom

            pydicom.dcmread(path, stop_before_pixels=True, force=False)
            detected = "dicom"
        except Exception:
            pass
    if detected not in ALLOWED_FORMATS[modality]:
        path.unlink(missing_ok=True)
        raise DownloadError(f"File format '{detected or 'unknown'}' is not allowed for {modality}.")

    try:
        yield path, detected
    finally:
        path.unlink(missing_ok=True)