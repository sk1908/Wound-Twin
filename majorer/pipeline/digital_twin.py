import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from config import OUTPUT_DIR
from .inference import PipelineResult


@dataclass
class WoundState:
    timestamp: str
    image_hash: str
    wound_type: str
    severity: str
    risk_level: str
    risk_score: float
    tissue_percentages: Dict[str, float]
    wound_area: float
    wound_volume: float
    mean_depth: float
    healthy_ratio: float
    necrotic_burden: float

    def to_dict(self) -> Dict:
        def safe_float(v):
            if v is None:
                return 0.0
            return float(v)

        def safe_dict(d):
            if d is None:
                return {}
            return {k: safe_float(v) for k, v in d.items()}

        return {
            "timestamp": self.timestamp,
            "image_hash": self.image_hash,
            "wound_type": str(self.wound_type),
            "severity": str(self.severity),
            "risk_level": str(self.risk_level),
            "risk_score": safe_float(self.risk_score),
            "tissue_percentages": safe_dict(self.tissue_percentages),
            "wound_area": safe_float(self.wound_area),
            "wound_volume": safe_float(self.wound_volume),
            "mean_depth": safe_float(self.mean_depth),
            "healthy_ratio": safe_float(self.healthy_ratio),
            "necrotic_burden": safe_float(self.necrotic_burden),
        }


class DigitalTwin:
    def __init__(
        self,
        patient_id: str = "default",
        wound_id: str = "wound_1",
        storage_dir: Optional[Path] = None,
    ):
        self.patient_id = patient_id
        self.wound_id = wound_id
        self.storage_dir = storage_dir or (OUTPUT_DIR / "digital_twins" / patient_id / wound_id)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.states: List[WoundState] = []
        self._load()

    def _load(self):
        history_file = self.storage_dir / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                data = json.load(f)
                for state_dict in data.get("states", []):
                    self.states.append(WoundState(**state_dict))
            print(f"Loaded {len(self.states)} historical states")

    def _save(self):
        history_file = self.storage_dir / "history.json"
        with open(history_file, "w") as f:
            json.dump(
                {
                    "patient_id": self.patient_id,
                    "wound_id": self.wound_id,
                    "states": [s.to_dict() for s in self.states],
                },
                f,
                indent=2,
            )

    def _compute_image_hash(self, image: np.ndarray) -> str:
        return hashlib.md5(image.tobytes()).hexdigest()[:16]

    def update(self, pipeline_result: PipelineResult, timestamp: Optional[str] = None) -> WoundState:
        timestamp = timestamp or datetime.now().isoformat()
        image_hash = self._compute_image_hash(pipeline_result.original_image)

        if self.states and self.states[-1].image_hash == image_hash:
            print("Duplicate image detected, skipping update")
            return self.states[-1]

        def to_float(v):
            if v is None:
                return 0.0
            return float(v)

        def to_dict_float(d):
            if d is None:
                return {}
            return {k: to_float(v) for k, v in d.items()}

        state = WoundState(
            timestamp=timestamp,
            image_hash=image_hash,
            wound_type=(
                str(pipeline_result.classification.wound_type)
                if pipeline_result.classification
                else "unknown"
            ),
            severity=(
                str(pipeline_result.classification.severity) if pipeline_result.classification else "unknown"
            ),
            risk_level=(str(pipeline_result.risk.risk_level) if pipeline_result.risk else "unknown"),
            risk_score=(to_float(pipeline_result.risk.risk_score) if pipeline_result.risk else 0.5),
            tissue_percentages=(
                to_dict_float(pipeline_result.segmentation.class_percentages)
                if pipeline_result.segmentation
                else {}
            ),
            wound_area=(to_float(pipeline_result.features.wound_area) if pipeline_result.features else 0.0),
            wound_volume=(
                to_float(pipeline_result.features.total_volume) if pipeline_result.features else 0.0
            ),
            mean_depth=(to_float(pipeline_result.features.mean_depth) if pipeline_result.features else 0.0),
            healthy_ratio=(
                to_float(pipeline_result.features.healthy_ratio) if pipeline_result.features else 0.0
            ),
            necrotic_burden=(
                to_float(pipeline_result.features.necrotic_burden) if pipeline_result.features else 0.0
            ),
        )

        self.states.append(state)
        self._save()

        print(f"Digital twin updated: {len(self.states)} states")

        return state

    def get_current_state(self) -> Optional[WoundState]:
        return self.states[-1] if self.states else None

    def get_history(self, last_n: Optional[int] = None) -> List[WoundState]:
        if last_n:
            return self.states[-last_n:]
        return self.states

    def compute_trends(self) -> Dict:
        if len(self.states) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 measurements",
            }

        recent = self.states[-5:]

        first = recent[0]
        last = recent[-1]

        area_change = last.wound_area - first.wound_area
        volume_change = last.wound_volume - first.wound_volume
        risk_change = last.risk_score - first.risk_score
        healthy_change = last.healthy_ratio - first.healthy_ratio

        if area_change < 0 and healthy_change > 0:
            overall = "improving"
        elif area_change > 0 or healthy_change < 0:
            overall = "worsening"
        else:
            overall = "stable"

        return {
            "status": "computed",
            "overall_trend": overall,
            "num_measurements": len(self.states),
            "area_change": area_change,
            "area_change_pct": ((area_change / first.wound_area * 100) if first.wound_area > 0 else 0),
            "volume_change": volume_change,
            "risk_change": risk_change,
            "healthy_ratio_change": healthy_change,
            "measurements": [s.to_dict() for s in recent],
        }

    def predict_healing_time(self) -> Dict:
        trends = self.compute_trends()

        if trends["status"] != "computed":
            return {"status": "insufficient_data"}

        current = self.get_current_state()
        if not current:
            return {"status": "no_current_state"}

        if trends["area_change_pct"] < 0:
            rate = abs(trends["area_change_pct"]) / len(self.states)
            if rate > 0:
                days_to_heal = (current.wound_area / rate) / 10
                confidence = min(rate * 2, 0.9)
            else:
                days_to_heal = float("inf")
                confidence = 0.1
        else:
            days_to_heal = float("inf")
            confidence = 0.0

        return {
            "status": "predicted",
            "estimated_days_to_heal": min(days_to_heal, 365),
            "confidence": confidence,
            "current_risk_level": current.risk_level,
            "trend": trends["overall_trend"],
            "note": "This is an estimate based on limited data",
        }

    def generate_report(self) -> str:
        current = self.get_current_state()
        trends = self.compute_trends()
        prediction = self.predict_healing_time()

        lines = [
            "=" * 50,
            "DIGITAL TWIN WOUND REPORT",
            "=" * 50,
            f"Patient ID: {self.patient_id}",
            f"Wound ID: {self.wound_id}",
            f"Total Measurements: {len(self.states)}",
            "",
        ]

        if current:
            lines.extend(
                [
                    "CURRENT STATE:",
                    f"  Timestamp: {current.timestamp}",
                    f"  Wound Type: {current.wound_type}",
                    f"  Severity: {current.severity}",
                    f"  Risk Level: {current.risk_level}",
                    f"  Risk Score: {current.risk_score:.2f}",
                    f"  Wound Area: {current.wound_area:.0f} px",
                    f"  Mean Depth: {current.mean_depth:.3f}",
                    f"  Healthy Ratio: {current.healthy_ratio:.1%}",
                    f"  Necrotic Burden: {current.necrotic_burden:.1%}",
                    "",
                ]
            )

        if trends["status"] == "computed":
            lines.extend(
                [
                    "TRENDS:",
                    f"  Overall: {trends['overall_trend'].upper()}",
                    f"  Area Change: {trends['area_change_pct']:.1f}%",
                    f"  Risk Change: {trends['risk_change']:+.2f}",
                    "",
                ]
            )

        if prediction["status"] == "predicted":
            lines.extend(
                [
                    "PREDICTION:",
                    f"  Estimated Days to Heal: {prediction['estimated_days_to_heal']:.0f}",
                    f"  Confidence: {prediction['confidence']:.1%}",
                    "",
                ]
            )

        lines.append("=" * 50)

        return "\n".join(lines)

    def export_timeline(self, output_path: Optional[Path] = None) -> Path:
        output_path = output_path or (self.storage_dir / "timeline.json")

        timeline = {
            "patient_id": self.patient_id,
            "wound_id": self.wound_id,
            "measurements": [],
        }

        for state in self.states:
            timeline["measurements"].append(
                {
                    "timestamp": state.timestamp,
                    "risk_score": state.risk_score,
                    "wound_area": state.wound_area,
                    "healthy_ratio": state.healthy_ratio,
                    "severity": state.severity,
                }
            )

        with open(output_path, "w") as f:
            json.dump(timeline, f, indent=2)

        return output_path


if __name__ == "__main__":
    twin = DigitalTwin(patient_id="test_patient", wound_id="wound_1")

    for i in range(5):
        state = WoundState(
            timestamp=f"2024-01-{i+1:02d}T10:00:00",
            image_hash=f"hash_{i}",
            wound_type="diabetic",
            severity="moderate" if i < 3 else "mild",
            risk_level="moderate" if i < 3 else "low",
            risk_score=0.5 - i * 0.08,
            tissue_percentages={
                "granulation": 0.4 + i * 0.1,
                "necrotic": 0.2 - i * 0.04,
            },
            wound_area=1000 - i * 100,
            wound_volume=500 - i * 50,
            mean_depth=0.3 - i * 0.03,
            healthy_ratio=0.5 + i * 0.1,
            necrotic_burden=0.3 - i * 0.05,
        )
        twin.states.append(state)

    print(twin.generate_report())
    print("\nTrends:", twin.compute_trends())
    print("\nPrediction:", twin.predict_healing_time())
