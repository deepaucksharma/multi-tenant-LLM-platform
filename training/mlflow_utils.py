"""
MLflow experiment tracking and model registry utilities.
Provides structured tracking for multi-tenant training runs.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

try:
    import mlflow
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Tracking will use local JSON fallback.")


class ExperimentTracker:
    """
    Unified experiment tracking that uses MLflow when available,
    falls back to local JSON logging otherwise.
    """

    def __init__(
        self,
        experiment_name: str = "multi-tenant-llm",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self._run_id: Optional[str] = None
        self._run_data: Dict[str, Any] = {}
        self._local_log_dir = Path("mlruns/local_logs")
        self._local_log_dir.mkdir(parents=True, exist_ok=True)
        self._use_mlflow = False

        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._use_mlflow = True
                logger.info(f"MLflow tracking at: {self.tracking_uri}")
            except Exception as e:
                logger.warning(f"MLflow connection failed: {e}. Using local fallback.")
                self._use_mlflow = False

    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new tracking run."""
        self._run_data = {
            "run_name": run_name,
            "started_at": datetime.utcnow().isoformat(),
            "tags": tags or {},
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

        if self._use_mlflow:
            try:
                run = mlflow.start_run(run_name=run_name, tags=tags)
                self._run_id = run.info.run_id
                logger.info(f"MLflow run started: {self._run_id}")
                return self._run_id
            except Exception as e:
                logger.warning(f"MLflow start_run failed: {e}")
                self._use_mlflow = False

        self._run_id = f"local_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{run_name}"
        logger.info(f"Local run started: {self._run_id}")
        return self._run_id

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        flat_params = self._flatten_dict(params)
        self._run_data["params"].update(flat_params)

        if self._use_mlflow:
            try:
                safe_params = {k: str(v)[:250] for k, v in flat_params.items()}
                mlflow.log_params(safe_params)
            except Exception as e:
                logger.warning(f"MLflow log_params failed: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        self._run_data["metrics"].update(metrics)

        if self._use_mlflow:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"MLflow log_metrics failed: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        self._run_data["artifacts"].append(str(path))

        if self._use_mlflow:
            try:
                mlflow.log_artifact(path, artifact_path)
            except Exception as e:
                logger.warning(f"MLflow log_artifact failed: {e}")

    def log_model_info(
        self,
        tenant_id: str,
        model_type: str,
        base_model: str,
        adapter_path: Optional[str] = None,
        metrics: Optional[Dict] = None,
    ):
        """Log structured model information for registry."""
        model_info = {
            "tenant_id": tenant_id,
            "model_type": model_type,
            "base_model": base_model,
            "adapter_path": adapter_path,
            "metrics": metrics or {},
            "registered_at": datetime.utcnow().isoformat(),
        }
        self._run_data["model_info"] = model_info

        if self._use_mlflow:
            try:
                mlflow.log_params({
                    "tenant_id": tenant_id,
                    "model_type": model_type,
                    "base_model": base_model,
                })
                if adapter_path:
                    mlflow.log_param("adapter_path", adapter_path)
            except Exception:
                pass

    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        self._run_data["ended_at"] = datetime.utcnow().isoformat()
        self._run_data["status"] = status

        log_path = self._local_log_dir / f"{self._run_id}.json"
        log_path.write_text(json.dumps(self._run_data, indent=2, default=str))

        if self._use_mlflow:
            try:
                mlflow.end_run(status=status)
            except Exception as e:
                logger.warning(f"MLflow end_run failed: {e}")

        logger.info(f"Run ended: {self._run_id} [{status}]")

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        """Flatten nested dict for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


class ModelRegistry:
    """
    Simple model registry that tracks adapter versions per tenant.
    Uses local JSON for persistence.
    """

    def __init__(self):
        self.registry_path = Path("models/registry.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self):
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": [], "active_versions": {}}

    def _save_registry(self):
        self.registry_path.write_text(
            json.dumps(self.registry, indent=2, default=str)
        )

    def register_model(
        self,
        tenant_id: str,
        model_type: str,
        version: str,
        adapter_path: str,
        base_model: str,
        metrics: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        dataset_info: Optional[Dict] = None,
        status: str = "staging",
    ) -> Dict:
        """Register a new model version."""
        entry = {
            "id": f"{tenant_id}_{model_type}_{version}",
            "tenant_id": tenant_id,
            "model_type": model_type,
            "version": version,
            "adapter_path": adapter_path,
            "base_model": base_model,
            "metrics": metrics or {},
            "training_config": training_config or {},
            "dataset_info": dataset_info or {},
            "status": status,
            "registered_at": datetime.utcnow().isoformat(),
            "promoted_at": None,
        }

        self.registry["models"].append(entry)
        self._save_registry()

        logger.info(
            f"Model registered: {entry['id']} | status={status} | "
            f"metrics={metrics}"
        )
        return entry

    def promote_to_production(self, tenant_id: str, model_type: str, version: str):
        """Promote a model version to production status."""
        model_id = f"{tenant_id}_{model_type}_{version}"
        current_key = f"{tenant_id}_{model_type}"
        current_prod = self.registry["active_versions"].get(current_key)

        if current_prod:
            for m in self.registry["models"]:
                if m["id"] == current_prod and m["status"] == "production":
                    m["status"] = "archived"

        for m in self.registry["models"]:
            if m["id"] == model_id:
                m["status"] = "production"
                m["promoted_at"] = datetime.utcnow().isoformat()
                self.registry["active_versions"][current_key] = model_id
                break

        self._save_registry()
        logger.info(f"Model promoted to production: {model_id}")

    def get_production_model(self, tenant_id: str, model_type: str = "sft") -> Optional[Dict]:
        """Get the current production model for a tenant."""
        current_key = f"{tenant_id}_{model_type}"
        model_id = self.registry["active_versions"].get(current_key)

        if model_id:
            for m in self.registry["models"]:
                if m["id"] == model_id:
                    return m

        tenant_models = [
            m for m in self.registry["models"]
            if m["tenant_id"] == tenant_id and m["model_type"] == model_type
        ]
        if tenant_models:
            return tenant_models[-1]

        return None

    def list_models(self, tenant_id: Optional[str] = None) -> List[Dict]:
        """List registered models, optionally filtered by tenant."""
        models = self.registry["models"]
        if tenant_id:
            models = [m for m in models if m["tenant_id"] == tenant_id]
        return models

    def get_summary(self) -> Dict:
        """Get registry summary."""
        return {
            "total_models": len(self.registry["models"]),
            "active_versions": self.registry["active_versions"],
            "by_tenant": {
                tid: len([m for m in self.registry["models"] if m["tenant_id"] == tid])
                for tid in set(m["tenant_id"] for m in self.registry["models"])
            } if self.registry["models"] else {},
        }
