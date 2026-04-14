"""Stage A training and calibration package.

Exports the LoRA training entrypoint and the FPR-constrained
threshold calibrator used by the policy gate.
"""

from src.training.calibrate import calibrate_thresholds
from src.training.train import train_stage_a

__all__ = ["train_stage_a", "calibrate_thresholds"]
