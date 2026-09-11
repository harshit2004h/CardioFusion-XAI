from src.extraction.schemas import DocumentExtraction, ExtractedTest
from src.normalization.biomarker_adapter import select_biomarker_input


def test_zheen_report_maps_to_exact_feature_order():
    extraction = DocumentExtraction(
        raw_text_available=True,
        tests=[
            ExtractedTest(canonical_candidate=name, source_label=name, value=float(index))
            for index, name in enumerate(
                ["age", "gender", "heart_rate", "systolic_bp", "diastolic_bp", "blood_glucose", "ck_mb", "troponin"]
            )
        ],
    )
    result = select_biomarker_input(extraction)
    assert result is not None
    assert result.source == "zheen"
    assert result.task == "acute_mi"
    assert result.values.tolist() == list(map(float, range(8)))
    assert result.missing_features == ()


def test_incomplete_biomarker_report_never_fabricates_missing_values():
    extraction = DocumentExtraction(
        raw_text_available=True,
        tests=[ExtractedTest(canonical_candidate="troponin", source_label="Troponin", value=0.8)],
    )
    result = select_biomarker_input(extraction)
    assert result is not None
    assert result.values.size == 0
    assert "age" in result.missing_features
