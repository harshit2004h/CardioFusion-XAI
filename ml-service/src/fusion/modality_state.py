from dataclasses import dataclass, field


@dataclass
class ModalityState:
    """
    Stores the currently available modalities
    and their disease-level predictions.
    """

    biomarkers: dict[str, float] = field(
        default_factory=dict
    )

    ecg: dict[str, float] = field(
        default_factory=dict
    )

    echo: dict[str, float] = field(
        default_factory=dict
    )

    def has_modality(
        self,
        modality: str,
    ) -> bool:
        values = getattr(
            self,
            modality,
            None
        )

        return bool(values)

    def available_modalities(self):
        modalities = []

        for modality in [
            "biomarkers",
            "ecg",
            "echo",
        ]:
            if self.has_modality(modality):
                modalities.append(modality)

        return modalities