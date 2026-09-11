from __future__ import annotations

import re
from pathlib import Path

from src.ingestion.dicom import read_document

from .schemas import DocumentExtraction, ExtractedTest

ALIASES = {
    "age": "age",
    "gender": "gender",
    "sex": "sex",
    "male": "sex",
    "heart_rate": "heart_rate",
    "systolic blood pressure": "systolic_bp",
    "systolic bp": "systolic_bp",
    "sysbp": "systolic_bp",
    "diastolic blood pressure": "diastolic_bp",
    "diastolic bp": "diastolic_bp",
    "diabp": "diastolic_bp",
    "blood sugar": "blood_glucose",
    "blood glucose": "blood_glucose",
    "glucose": "glucose",
    "education": "education",
    "cigarettes per day": "cigarettes_per_day",
    "cigsperday": "cigarettes_per_day",
    "current smoker": "current_smoker",
    "currentsmoker": "current_smoker",
    "bp meds": "bp_medication",
    "bpmeds": "bp_medication",
    "prevalent stroke": "prevalent_stroke",
    "prevalentstroke": "prevalent_stroke",
    "prevalent hypertension": "prevalent_hypertension",
    "prevalenthyp": "prevalent_hypertension",
    "diabetes": "diabetes",
    "total cholesterol": "total_cholesterol",
    "totchol": "total_cholesterol",
    "bmi": "bmi",
    "heart rate": "heart_rate",
    "anaemia": "anaemia",
    "anemia": "anaemia",
    "creatinine phosphokinase": "creatinine_phosphokinase",
    "ejection fraction": "ejection_fraction",
    "high blood pressure": "high_blood_pressure",
    "platelets": "platelets",
    "serum creatinine": "serum_creatinine",
    "smoking": "smoking",
    "troponin i": "troponin",
    "troponin-i": "troponin",
    "ctni": "troponin",
    "troponin": "troponin",
    "ck-mb": "ck_mb",
    "ck mb": "ck_mb",
    "sodium": "serum_sodium",
    "serum sodium": "serum_sodium",
    "na+": "serum_sodium",
    "creatinine": "serum_creatinine",
    "ef": "ejection_fraction",
}

_TEST_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9 +_./()-]{1,48})\s*[:=]?\s*"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z%µ/]+)?",
    re.IGNORECASE,
)


def _read_text(path: Path, file_format: str) -> str:
    if file_format == "pdf":
        try:
            from pypdf import PdfReader

            return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        except Exception:
            return ""
    if file_format in {"jpg", "jpeg", "png"}:
        try:
            import pytesseract
            from PIL import Image

            return pytesseract.image_to_string(Image.open(path))
        except Exception:
            return ""
    return ""


def extract_document(path: Path, file_format: str) -> DocumentExtraction:
    text = read_document(path).text if file_format == "dicom" else _read_text(path, file_format)
    tests: list[ExtractedTest] = []
    for match in _TEST_RE.finditer(text):
        label = " ".join(match.group("label").split()).strip(" -_:")
        canonical = ALIASES.get(label.casefold())
        tests.append(
            ExtractedTest(
                canonical_candidate=canonical,
                source_label=label,
                value=float(match.group("value")),
                unit=match.group("unit"),
                confidence=0.85 if canonical else 0.50,
            )
        )
    known = {test.canonical_candidate for test in tests if test.canonical_candidate}
    return DocumentExtraction(
        raw_text_available=bool(text.strip()),
        tests=tests,
        ignored_extra_tests=[test.source_label for test in tests if not test.canonical_candidate],
        extraction_quality=0.85 if tests else (0.25 if text.strip() else 0.0),
        missing_required_features=[] if known else ["configured biomarker features"],
    )