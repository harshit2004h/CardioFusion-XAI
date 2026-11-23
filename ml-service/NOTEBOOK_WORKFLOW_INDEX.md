# Notebook workflow and naming index

The added EDA, preprocessing, training, validation, calibration fitting, evaluation and XAI workflows run from notebooks. Existing `src/training/train_*.py` entrypoints are retained in this checkout; reusable model and data helpers live under `src/`.

> The historic lists below are retained as background only. Use this canonical
> order and the filenames present in the workspace.

## Canonical execution order

1. `01_eda/01_global_dataset_quality_audit_eda.ipynb` through `07_echonet_dynamic_eda.ipynb`
2. `02_preprocessing/01_biomarker_sources_preprocessing.ipynb` through `06_preprocessing_artifact_validation.ipynb`
3. Training branches, in their numbered order:
   - `03_training/biomarkers/01_multisource_biomarker_baseline_training.ipynb` through `04_biomarker_selected_gbdt_stacking_training.ipynb`
   - `03_training/ecg/01_ptbxl_cnn_1d_training.ipynb` through `05_ptbxl_selected_cnn_bilstm_training.ipynb`
   - `03_training/echo/01_echonet_r2plus1d_training.ipynb` through `04_panecho_selected_finetuning_training.ipynb`
4. Evaluation branches, in their numbered order under `04_evaluation/`.
5. `05_calibration_fusion/01_legacy_calibration_fusion.ipynb` through `06_disease_evidence_validation_fusion.ipynb`; the active service calibration is `03_selected_models_calibration_fusion.ipynb`.
6. `06_explainability/01_biomarker_shap_explainability.ipynb` through `03_echo_video_explainability.ipynb`.

## Canonical model and artifact layout

Every selected model is written to:

```text
checkpoints/<modality>/<dataset>/<model>/v1/model.<extension>
```

Metrics, prediction exports, split manifests and plots are written to:

```text
artifacts/<stage>/<modality>/<dataset>/<model>/v1/
```

The active service reads:

```text
checkpoints/biomarkers/<dataset>/<target>/selected_gbdt_stacking/v1/model.joblib
checkpoints/ecg/ptbxl/selected_cnn_bilstm/v1/model.pt
checkpoints/echo/echonet/selected_panecho/v1/model.pt
checkpoints/calibration/selected_models/v1/manifest.json
```

## Original notebooks

Every notebook supplied in the original repository ZIP is preserved byte-for-byte. In particular, do not overwrite these original experiment notebooks when adding further variants:

- Biomarkers: `03_training/biomarkers/01_multisource_biomarker_baseline_training.ipynb`, `02_cardiac_biomarker_classical_training.ipynb`
- ECG: `03_training/ecg/01_ptbxl_cnn_1d_training.ipynb`, `02_ptbxl_cnn_bilstm_training.ipynb`, `03_ptbxl_cnn_resnet_training.ipynb`
- Echo: `03_training/echo/01_echonet_r2plus1d_training.ipynb`, `02_panecho_baseline_training.ipynb`, `03_panecho_finetuning_training.ipynb`
- Original preprocessing, EDA, evaluation and fusion notebooks remain in their original directories.

## Additional notebooks

| Notebook | Purpose |
|---|---|
| `01_eda/01_global_dataset_quality_audit_eda.ipynb` | Cross-dataset schema and target audit without joining patients |
| `02_preprocessing/06_preprocessing_artifact_validation.ipynb` | Check processed data, manifests and split contracts |
| `03_training/biomarkers/03_biomarker_ft_transformer_training.ipynb` | Four independent source-specific FT-Transformer biomarker/clinical experts |
| `03_training/ecg/04_ptbxl_cnn_transformer_training.ipynb` | Additional CNN + temporal Transformer ECG experiment |
| `04_evaluation/biomarkers/03_biomarker_ft_transformer_evaluation.ipynb` | Held-out evaluation of the added FT-Transformer experts |
| `04_evaluation/ecg/04_ptbxl_model_comparison_evaluation.ipynb` | Compare saved CNN-BiLSTM and CNN-Transformer outputs with aggregation caveats |
| `05_calibration_fusion/02_experiment_calibration_validation.ipynb` | Fit task-specific calibration from validation predictions |
| `05_calibration_fusion/06_disease_evidence_validation_fusion.ipynb` | Validate disease routing and all seven upload combinations |
| `06_explainability/01_biomarker_shap_explainability.ipynb` | Clinical/biomarker SHAP analyses |
| `06_explainability/02_ecg_signal_explainability.ipynb` | ECG signal explanation analyses |
| `06_explainability/03_echo_video_explainability.ipynb` | Echo temporal/spatial explanation analyses |
| `01_eda/03_uci_heart_failure_eda.ipynb` | Full cohort EDA; analyzes follow-up time but excludes it from predictors |
| `03_training/biomarkers/04_biomarker_selected_gbdt_stacking_training.ipynb` | Independent dataset-specific GBDT/stacking selection with reserved calibration and test splits |
| `03_training/ecg/05_ptbxl_selected_cnn_bilstm_training.ipynb` | Existing CNN-BiLSTM architecture with patient-level internal model-selection holdout and reserved official validation predictions |
| `03_training/echo/04_panecho_selected_finetuning_training.ipynb` | PanEcho fine-tuning with internal holdout and reserved validation EF predictions |
| `05_calibration_fusion/03_selected_models_calibration_fusion.ipynb` | Platt calibration for supported selected classification targets and split-conformal EF intervals |

## Execution order

1. Run the numbered EDA and preprocessing notebooks. Run `01_eda/03_uci_heart_failure_eda.ipynb` for the UCI EDA slot.
2. Run the original model training notebooks to reproduce existing baselines.
3. Run the additional FT-Transformer biomarker notebook and CNN-Transformer ECG notebook for comparative experiments.
4. Run the corresponding evaluation notebooks on the held-out test splits.
5. For selected-model calibration, run the three selected-model training notebooks independently, then run `05_calibration_fusion/03_selected_models_calibration_fusion.ipynb`. Use `02_experiment_calibration_validation.ipynb` for optional experiment calibration.
6. Run the disease-evidence orchestration and XAI notebooks.

Do not train or evaluate a cross-modal predictor by pairing unrelated rows from the six independent datasets.

The new calibration-ready notebooks write to separate checkpoint and artifact paths. They do not overwrite original experiment outputs. Biomarker and ECG classification use Platt scaling only when the calibration fold has at least 30 rows and at least 10 examples from each class; otherwise the target remains uncalibrated. The small UCI heart-failure mortality cohort is always left uncalibrated. Platt artifacts and EF intervals are loaded only when their manifest SHA-256 matches the exact inference checkpoint. Echo EF uses absolute-residual split conformal calibration and reports marginal coverage under exchangeability, not patient-specific certainty. The service does not report uncalibrated or unverified classifier scores as patient risk.
