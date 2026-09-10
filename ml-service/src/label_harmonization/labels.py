# Canonical Taxonomy: The single source of truth for all mapped targets
CANONICAL_DISEASES = [
    "acute_mi", "myocardial_ischemia", "st_t_abnormalities",
    "atrial_fibrillation", "bradyarrhythmia", "av_block",
    "lbbb_rbbb", "conduction_abnormalities", "lvh_rvh",
    "lv_dysfunction", "reduced_lvef", "hfref",
    "hf_mortality", "cardiogenic_shock", "acute_lv_failure",
    "recurrent_mi", "post_mi_arrhythmias", "post_mi_mortality",
    "10_year_chd_risk"
]

# Tabular Dataset Label Mappings
BIOMARKER_LABEL_MAP = {
    "zheen": {
        "class": "acute_mi"  
    },
    "uci_hf": {
        "DEATH_EVENT": "hf_mortality"
    },
    "mi_complications": {
        "K_SH_POST": "cardiogenic_shock",
        "O_L_POST": "acute_lv_failure",
        "OTEK_LANC": "acute_lv_failure",
        "REC_IM": "recurrent_mi",
        "FIBR_PREDS": "post_mi_arrhythmias", # AF
        "PREDS_TAH": "post_mi_arrhythmias",  # SVT
        "JELUD_TAH": "post_mi_arrhythmias",  # VT
        "FIBR_JELUD": "post_mi_arrhythmias", # VF
        "A_V_BLOK": "post_mi_arrhythmias",   # 3-degree AV block
        "LET_IS": "post_mi_mortality"
    },
    "framingham": {
        "TenYearCHD": "10_year_chd_risk"
    }
}

# PTB-XL SCP-ECG Statement Mappings
ECG_SCP_MAP = {
    "MI": "acute_mi",
    "ISC_": "myocardial_ischemia",
    "STTC": "st_t_abnormalities",
    "AFIB": "atrial_fibrillation",
    "AFLT": "atrial_fibrillation",
    "SBRAD": "bradyarrhythmia",
    "_AVB": "av_block",
    "CLBBB": "lbbb_rbbb",
    "ILBBB": "lbbb_rbbb",
    "CRBBB": "lbbb_rbbb",
    "IRBBB": "lbbb_rbbb",
    "HYP": "lvh_rvh"
}

def get_echonet_targets(ef_value: float) -> dict:
    """Derives discrete structural diseases from the continuous LVEF regression label."""
    return {
        "reduced_lvef": ef_value,
        "lv_dysfunction": int(ef_value < 50.0),
        "hfref": int(ef_value < 40.0)
    }