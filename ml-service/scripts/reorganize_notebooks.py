"""Apply the canonical notebook naming and stage-directory layout once.

The mapping is intentionally explicit so filename changes are reviewable and
references in notebooks, docs and source files are migrated together.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_MOVES = {
    "notebooks/01_eda/six_dataset_quality_and_target_audit.ipynb": "notebooks/01_eda/01_global_dataset_quality_audit_eda.ipynb",
    "notebooks/01_eda/zheen.ipynb": "notebooks/01_eda/02_zheen_biomarker_eda.ipynb",
    "notebooks/01_eda/uci_heart_failure_eda.ipynb": "notebooks/01_eda/03_uci_heart_failure_eda.ipynb",
    "notebooks/01_eda/framingham.ipynb": "notebooks/01_eda/04_framingham_chd_eda.ipynb",
    "notebooks/01_eda/mi_complications.ipynb": "notebooks/01_eda/05_myocardial_infarction_complications_eda.ipynb",
    "notebooks/01_eda/ptbxl.ipynb": "notebooks/01_eda/06_ptbxl_ecg_eda.ipynb",
    "notebooks/01_eda/echonet.ipynb": "notebooks/01_eda/07_echonet_dynamic_eda.ipynb",
    "notebooks/02_preprocessing/biomarkers.ipynb": "notebooks/02_preprocessing/01_biomarker_sources_preprocessing.ipynb",
    "notebooks/02_preprocessing/ptbxl.ipynb": "notebooks/02_preprocessing/02_ptbxl_ecg_preprocessing.ipynb",
    "notebooks/02_preprocessing/echonet.ipynb": "notebooks/02_preprocessing/03_echonet_dynamic_preprocessing.ipynb",
    "notebooks/02_preprocessing/panecho.ipynb": "notebooks/02_preprocessing/04_panecho_baseline_preprocessing.ipynb",
    "notebooks/02_preprocessing/panecho_preprocessing_updated.ipynb": "notebooks/02_preprocessing/05_panecho_updated_preprocessing.ipynb",
    "notebooks/02_preprocessing/preprocessing_artifact_validation.ipynb": "notebooks/02_preprocessing/06_preprocessing_artifact_validation.ipynb",
    "notebooks/03_training/biomarkers/biomarkers.ipynb": "notebooks/03_training/biomarkers/01_multisource_biomarker_baseline_training.ipynb",
    "notebooks/03_training/biomarkers/cardiac_biomarker_training.ipynb": "notebooks/03_training/biomarkers/02_cardiac_biomarker_classical_training.ipynb",
    "notebooks/03_training/biomarkers/cardiac_biomarker_ft_transformer.ipynb": "notebooks/03_training/biomarkers/03_biomarker_ft_transformer_training.ipynb",
    "notebooks/03_training/biomarkers/cardiac_biomarker_calibration_ready.ipynb": "notebooks/03_training/biomarkers/04_biomarker_selected_gbdt_stacking_training.ipynb",
    "notebooks/03_training/ecg/ptbxl_cnn_1D.ipynb": "notebooks/03_training/ecg/01_ptbxl_cnn_1d_training.ipynb",
    "notebooks/03_training/ecg/ptbxl_cnn_BiLSTM.ipynb": "notebooks/03_training/ecg/02_ptbxl_cnn_bilstm_training.ipynb",
    "notebooks/03_training/ecg/ptbxl_cnn_resnet.ipynb": "notebooks/03_training/ecg/03_ptbxl_cnn_resnet_training.ipynb",
    "notebooks/03_training/ecg/ptbxl_cnn_transformer.ipynb": "notebooks/03_training/ecg/04_ptbxl_cnn_transformer_training.ipynb",
    "notebooks/03_training/ecg/ptbxl_cnn_BiLSTM_calibration_ready.ipynb": "notebooks/03_training/ecg/05_ptbxl_selected_cnn_bilstm_training.ipynb",
    "notebooks/03_training/echo/echonet.ipynb": "notebooks/03_training/echo/01_echonet_r2plus1d_training.ipynb",
    "notebooks/03_training/echo/panecho.ipynb": "notebooks/03_training/echo/02_panecho_baseline_training.ipynb",
    "notebooks/03_training/echo/panecho_finetuning_updated.ipynb": "notebooks/03_training/echo/03_panecho_finetuning_training.ipynb",
    "notebooks/03_training/echo/panecho_finetuning_calibration_ready.ipynb": "notebooks/03_training/echo/04_panecho_selected_finetuning_training.ipynb",
    "notebooks/04_evaluation/biomarkers/biomarkers.ipynb": "notebooks/04_evaluation/biomarkers/01_multisource_biomarker_baseline_evaluation.ipynb",
    "notebooks/04_evaluation/biomarkers/cardiac_biomarker_evaluation.ipynb": "notebooks/04_evaluation/biomarkers/02_cardiac_biomarker_classical_evaluation.ipynb",
    "notebooks/04_evaluation/biomarkers/cardiac_biomarker_ft_transformer_evaluation.ipynb": "notebooks/04_evaluation/biomarkers/03_biomarker_ft_transformer_evaluation.ipynb",
    "notebooks/04_evaluation/ecg/ptbxl_cnn_1D.ipynb": "notebooks/04_evaluation/ecg/01_ptbxl_cnn_1d_evaluation.ipynb",
    "notebooks/04_evaluation/ecg/ptbxl_cnn_BiLSTM.ipynb": "notebooks/04_evaluation/ecg/02_ptbxl_cnn_bilstm_evaluation.ipynb",
    "notebooks/04_evaluation/ecg/ptbxl_cnn_resnet.ipynb": "notebooks/04_evaluation/ecg/03_ptbxl_cnn_resnet_evaluation.ipynb",
    "notebooks/04_evaluation/ecg/ptbxl_ecg_model_comparison.ipynb": "notebooks/04_evaluation/ecg/04_ptbxl_model_comparison_evaluation.ipynb",
    "notebooks/04_evaluation/echo/echonet.ipynb": "notebooks/04_evaluation/echo/01_echonet_r2plus1d_evaluation.ipynb",
    "notebooks/04_evaluation/echo/panecho.ipynb": "notebooks/04_evaluation/echo/02_panecho_baseline_evaluation.ipynb",
    "notebooks/04_evaluation/echo/panecho_evaluation_updated.ipynb": "notebooks/04_evaluation/echo/03_panecho_finetuning_evaluation.ipynb",
    "notebooks/05_fusion/calibration.ipynb": "notebooks/05_calibration_fusion/01_legacy_calibration_fusion.ipynb",
    "notebooks/05_fusion/calibration_fitting_and_validation.ipynb": "notebooks/05_calibration_fusion/02_experiment_calibration_validation.ipynb",
    "notebooks/05_fusion/calibration_selected_models.ipynb": "notebooks/05_calibration_fusion/03_selected_models_calibration_fusion.ipynb",
    "notebooks/05_fusion/fusion_evaluation.ipynb": "notebooks/05_calibration_fusion/04_fusion_evaluation.ipynb",
    "notebooks/05_fusion/staged_inference.ipynb": "notebooks/05_calibration_fusion/05_staged_inference_fusion.ipynb",
    "notebooks/05_fusion/disease_evidence_orchestration_validation.ipynb": "notebooks/05_calibration_fusion/06_disease_evidence_validation_fusion.ipynb",
    "notebooks/06_xai/biomarker_shap.ipynb": "notebooks/06_explainability/01_biomarker_shap_explainability.ipynb",
    "notebooks/06_xai/ecg_explainability.ipynb": "notebooks/06_explainability/02_ecg_signal_explainability.ipynb",
    "notebooks/06_xai/echo_explainability.ipynb": "notebooks/06_explainability/03_echo_video_explainability.ipynb",
}
DIRECTORY_MOVES = {
    "notebooks/05_fusion": "notebooks/05_calibration_fusion",
    "notebooks/06_xai": "notebooks/06_explainability",
}
TEXT_EXTENSIONS = {".ipynb", ".md", ".py", ".yaml", ".yml", ".toml", ".txt", ".example"}
SKIP_PARTS = {".git", ".venv-cardio", ".pytest_cache", ".ruff_cache", "node_modules", "__pycache__"}


def replacements() -> list[tuple[str, str]]:
    pairs = [*NOTEBOOK_MOVES.items(), *DIRECTORY_MOVES.items()]
    values: list[tuple[str, str]] = []
    for old, new in pairs:
        values.append((old, new))
        values.append((old.replace("/", "\\"), new.replace("/", "\\")))
    return sorted(values, key=lambda pair: len(pair[0]), reverse=True)


def rewrite_references() -> int:
    changed = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path == Path(__file__).resolve():
            continue
        if path.suffix not in TEXT_EXTENSIONS and path.name != ".env.example":
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = original
        for old, new in replacements():
            updated = updated.replace(old, new)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def move_notebooks() -> None:
    missing = [old for old in NOTEBOOK_MOVES if not (ROOT / old).is_file()]
    occupied = [new for new in NOTEBOOK_MOVES.values() if (ROOT / new).exists()]
    if missing or occupied:
        raise RuntimeError(f"Refusing partial migration; missing={missing}, occupied={occupied}")
    for old, new in NOTEBOOK_MOVES.items():
        destination = ROOT / new
        destination.parent.mkdir(parents=True, exist_ok=True)
        (ROOT / old).rename(destination)
    for old in DIRECTORY_MOVES:
        directory = ROOT / old
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()


if __name__ == "__main__":
    move_notebooks()
    print(f"Moved {len(NOTEBOOK_MOVES)} notebooks and updated {rewrite_references()} text files.")
