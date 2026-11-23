"""Normalize the selected-model notebooks to the canonical output layout."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLACEMENTS = {
    "root/'artifacts/metrics/biomarkers/cardiac_biomarker_calibration_ready'": "root/'artifacts/03_training/biomarkers/selected_gbdt_stacking/v1'",
    "root/'checkpoints/biomarkers/calibration_ready'": "root/'checkpoints/biomarkers'",
    "model_path=model_dir/f'{source}_{target_name}.joblib'": "model_path=model_dir/source/'selected_gbdt_stacking'/'v1'/'model.joblib';model_path.parent.mkdir(parents=True,exist_ok=True)",
    "PROJECT_ROOT/'checkpoints/ecg/cnn_bilstm_attention_calibration_ready'": "PROJECT_ROOT/'checkpoints/ecg/ptbxl/selected_cnn_bilstm/v1'",
    "PROJECT_ROOT/'artifacts/metrics/ecg/cnn_bilstm_attention_calibration_ready'": "PROJECT_ROOT/'artifacts/03_training/ecg/ptbxl/selected_cnn_bilstm/v1'",
    "f\"ptbxl_cnn_bilstm_attention_calibration_ready_\"\n    f\"{PREPROCESSING_VERSION}.pt\"": "'model.pt'",
    "PROJECT_ROOT/'checkpoints/echo/panecho.pt'": "PROJECT_ROOT/'checkpoints/echo/panecho/pretrained/v1/model.pt'",
    "PROJECT_ROOT/'checkpoints/echo/panecho/panecho_echonet_calibration_ready.pt'": "PROJECT_ROOT/'checkpoints/echo/echonet/selected_panecho/v1/model.pt'",
    "PROJECT_ROOT/'artifacts/metrics/echo/panecho/calibration'": "PROJECT_ROOT/'artifacts/03_training/echo/echonet/selected_panecho/v1'",
    "root/'artifacts/metrics/calibration_selected_models'": "root/'artifacts/05_calibration_fusion/selected_models/v1'",
    "root/'checkpoints/calibration'": "root/'checkpoints/calibration/selected_models/v1'",
    "(ART/'selected_models_manifest.json')": "(ART/'manifest.json')",
    "root/'artifacts/metrics/biomarkers/cardiac_biomarker_calibration_ready'": "root/'artifacts/03_training/biomarkers/selected_gbdt_stacking/v1'",
    "root/'checkpoints/biomarkers/calibration_ready'/f'{dataset}_{target}.joblib'": "root/'checkpoints/biomarkers'/dataset/'selected_gbdt_stacking'/'v1'/'model.joblib'",
    "root/'artifacts/metrics/ecg/cnn_bilstm_attention_calibration_ready'": "root/'artifacts/03_training/ecg/ptbxl/selected_cnn_bilstm/v1'",
    "root/'checkpoints/ecg/cnn_bilstm_attention_calibration_ready/ptbxl_cnn_bilstm_attention_calibration_ready_100hz_record_zscore.pt'": "root/'checkpoints/ecg/ptbxl/selected_cnn_bilstm/v1/model.pt'",
    "root/'artifacts/metrics/echo/panecho/calibration'": "root/'artifacts/03_training/echo/echonet/selected_panecho/v1'",
    "root/'checkpoints/echo/panecho/panecho_echonet_calibration_ready.pt'": "root/'checkpoints/echo/echonet/selected_panecho/v1/model.pt'",
    "ART/'biomarkers_selected'/dataset/f'{target}.pkl'": "ART/'biomarkers'/dataset/'selected_gbdt_stacking'/f'{target}.pkl'",
    "ART/'ecg'/'cnn_bilstm_attention'/f'{target}.pkl'": "ART/'ecg'/'ptbxl'/'selected_cnn_bilstm'/f'{target}.pkl'",
    "ART/'echo/ef_absolute_error_quantile.json'": "ART/'echo'/'echonet'/'selected_panecho'/'ef_absolute_error_quantile.json'",
    "path.relative_to(ART).as_posix()": "path.relative_to(root/'checkpoints/calibration').as_posix()",
    "dest.relative_to(ART).as_posix()": "dest.relative_to(root/'checkpoints/calibration').as_posix()",
    "model_path=model_dir/source/'selected_gbdt_stacking'/'v1'/f'{target_name}.joblib'": "model_path=model_dir/source/target_name/'selected_gbdt_stacking'/'v1'/'model.joblib'",
    "base=root/'checkpoints/biomarkers'/dataset/'selected_gbdt_stacking'/'v1'/f'{target}.joblib'": "base=root/'checkpoints/biomarkers'/dataset/target/'selected_gbdt_stacking'/'v1'/'model.joblib'",
}
TARGETS = (
    "notebooks/03_training/biomarkers/04_biomarker_selected_gbdt_stacking_training.ipynb",
    "notebooks/03_training/ecg/05_ptbxl_selected_cnn_bilstm_training.ipynb",
    "notebooks/03_training/echo/04_panecho_selected_finetuning_training.ipynb",
    "notebooks/05_calibration_fusion/03_selected_models_calibration_fusion.ipynb",
)


def rewrite_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        updated = source
        for old, new in REPLACEMENTS.items():
            updated = updated.replace(old, new)
        if updated != source:
            cell["source"] = updated.splitlines(keepends=True)
            changed = True
    if not changed:
        return False
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


if __name__ == "__main__":
    changed = sum(rewrite_notebook(ROOT / relative_path) for relative_path in TARGETS)
    print(f"Normalized output paths in {changed} selected-model notebooks.")
