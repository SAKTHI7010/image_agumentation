"""Phase 4: Documented Project 3 Flotation Kinetic Model Integration Adapter.

No Project 3 kinetic model, source code, or input schema was supplied in the workspace.
Per specification, this module implements a typed, documented integration interface
and explicitly marks live recovery comparison as pending until Project 3 is connected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass
class Project3OperatingInputs:
    """Baseline flotation operating variables typically consumed by kinetic models."""

    air_flow_l_min: float | None = None
    collector_dosage_gt: float | None = None
    frother_condition_gt: float | None = None
    kind_of_water: str | None = None
    mineral_used: str | None = None
    residence_time_min: float | None = None
    impeller_rpm: float | None = None


@dataclass
class Project3ImageFrothFeatures:
    """Image-derived froth stability and visual descriptors exported by Project 7."""

    predicted_stability_label: str
    predicted_unstable_prob: float
    visual_stability_proxy: float
    bubble_d50_px: float | None
    bubble_size_span: float | None
    texture_entropy: float
    dark_fraction: float
    novelty_score: float
    drift_warning: bool


class FlotationKineticModelProtocol(Protocol):
    """Protocol that an external Project 3 flotation kinetic model must implement."""

    def predict_recovery(self, features: dict[str, Any]) -> dict[str, float]:
        """Return predicted cumulative recovery and kinetic parameters (e.g. R_inf, k)."""
        ...


class Project3KineticModelAdapter:
    """Adapter bridging Project 7 froth stability outputs to Project 3 kinetic models.

    When a Project 3 model is attached via `model=...`, this adapter prepares both:
      1) Operating-variables-only feature payload
      2) Operating-variables + image-derived froth stability payload
    so Project 3 can compare recovery prediction accuracy with vs. without froth vision.
    """

    MISSING_REQUIREMENTS = [
        "Project 3 kinetic model executable/module or serialized artifact (e.g., classical first-order / Kelsall / Klimpel model or ML regressor)",
        "Confirmed Project 3 input feature schema and physical units (e.g., residence_time_min, k_fast, k_slow, R_inf, feed_grade)",
        "Paired plant/lab flotation recovery and concentrate-grade measurements aligned with froth image timestamps for A/B validation",
    ]

    def __init__(self, model: FlotationKineticModelProtocol | None = None) -> None:
        self.model = model

    @property
    def is_connected(self) -> bool:
        return self.model is not None

    def build_comparison_payload(
        self,
        operating: Project3OperatingInputs,
        image_features: Project3ImageFrothFeatures,
    ) -> dict[str, Any]:
        """Construct structured baseline vs image-augmented payloads and invoke Project 3 if connected."""
        operating_only = asdict(operating)
        image_dict = asdict(image_features)
        operating_plus_image = {
            **operating_only,
            "froth_stability_score": round(1.0 - image_features.predicted_unstable_prob, 4),
            **image_dict,
        }

        if self.model is None:
            return {
                "integration_status": "pending_project3_model",
                "model_connected": False,
                "message": (
                    "Project 3 kinetic model is not supplied in this workspace. "
                    "Image-derived froth stability payload is ready and documented; "
                    "attach a Project 3 model implementing FlotationKineticModelProtocol to run live recovery comparison."
                ),
                "operating_only_payload": operating_only,
                "operating_plus_image_payload": operating_plus_image,
                "recovery_operating_only": None,
                "recovery_operating_plus_image": None,
                "missing_requirements": self.MISSING_REQUIREMENTS,
            }

        rec_base = self.model.predict_recovery(operating_only)
        rec_aug = self.model.predict_recovery(operating_plus_image)
        return {
            "integration_status": "connected",
            "model_connected": True,
            "operating_only_payload": operating_only,
            "operating_plus_image_payload": operating_plus_image,
            "recovery_operating_only": rec_base,
            "recovery_operating_plus_image": rec_aug,
            "missing_requirements": [],
        }
