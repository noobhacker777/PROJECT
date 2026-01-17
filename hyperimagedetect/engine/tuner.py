
from __future__ import annotations

import gc
import random
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np
import torch

from hyperimagedetect.cfg import get_cfg, get_save_dir
from hyperimagedetect.utils import DEFAULT_CFG, LOGGER, YAML, callbacks, colorstr, remove_colorstr
from hyperimagedetect.utils.checks import check_requirements
from hyperimagedetect.utils.patches import torch_load
from hyperimagedetect.utils.plotting import plot_tune_results

class Tuner:

    def __init__(self, args=DEFAULT_CFG, _callbacks: list | None = None):
        self.space = args.pop("space", None) or {
            "lr0": (1e-5, 1e-1),
            "lrf": (0.0001, 0.1),
            "momentum": (0.7, 0.98, 0.3),
            "weight_decay": (0.0, 0.001),
            "warmup_epochs": (0.0, 5.0),
            "warmup_momentum": (0.0, 0.95),
            "box": (1.0, 20.0),
            "cls": (0.1, 4.0),
            "dfl": (0.4, 6.0),
            "hsv_h": (0.0, 0.1),
            "hsv_s": (0.0, 0.9),
            "hsv_v": (0.0, 0.9),
            "degrees": (0.0, 45.0),
            "translate": (0.0, 0.9),
            "scale": (0.0, 0.95),
            "shear": (0.0, 10.0),
            "perspective": (0.0, 0.001),
            "flipud": (0.0, 1.0),
            "fliplr": (0.0, 1.0),
            "bgr": (0.0, 1.0),
            "mosaic": (0.0, 1.0),
            "mixup": (0.0, 1.0),
            "cutmix": (0.0, 1.0),
            "copy_paste": (0.0, 1.0),
            "close_mosaic": (0.0, 10.0),
        }
        mongodb_uri = args.pop("mongodb_uri", None)
        mongodb_db = args.pop("mongodb_db", "hyperimagedetect")
        mongodb_collection = args.pop("mongodb_collection", "tuner_results")

        self.args = get_cfg(overrides=args)
        self.args.exist_ok = self.args.resume
        self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")
        self.args.name, self.args.exist_ok, self.args.resume = (None, False, False)
        self.tune_csv = self.tune_dir / "tune_results.csv"
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.prefix = colorstr("Tuner: ")
        callbacks.add_integration_callbacks(self)

        self.mongodb = None
        if mongodb_uri:
            self._init_mongodb(mongodb_uri, mongodb_db, mongodb_collection)

        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
            f"{self.prefix}💡 Learn about tuning at https://docs.hyperimagedetect.com/guides/hyperparameter-tuning"
        )

    def _connect(self, uri: str = "mongodb+srv://username:password@cluster.mongodb.net/", max_retries: int = 3):
        check_requirements("pymongo")

        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        for attempt in range(max_retries):
            try:
                client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=20000,
                    socketTimeoutMS=40000,
                    retryWrites=True,
                    retryReads=True,
                    maxPoolSize=30,
                    minPoolSize=3,
                    maxIdleTimeMS=60000,
                )
                client.admin.command("ping")
                LOGGER.info(f"{self.prefix}Connected to MongoDB Atlas (attempt {attempt + 1})")
                return client
            except (ConnectionFailure, ServerSelectionTimeoutError):
                if attempt == max_retries - 1:
                    raise
                wait_time = 2**attempt
                LOGGER.warning(
                    f"{self.prefix}MongoDB connection failed (attempt {attempt + 1}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

    def _init_mongodb(self, mongodb_uri="", mongodb_db="", mongodb_collection=""):
        self.mongodb = self._connect(mongodb_uri)
        self.collection = self.mongodb[mongodb_db][mongodb_collection]
        self.collection.create_index([("fitness", -1)], background=True)
        LOGGER.info(f"{self.prefix}Using MongoDB Atlas for distributed tuning")

    def _get_mongodb_results(self, n: int = 5) -> list:
        try:
            return list(self.collection.find().sort("fitness", -1).limit(n))
        except Exception:
            return []

    def _save_to_mongodb(self, fitness: float, hyperparameters: dict[str, float], metrics: dict, iteration: int):
        try:
            self.collection.insert_one(
                {
                    "fitness": fitness,
                    "hyperparameters": {k: (v.item() if hasattr(v, "item") else v) for k, v in hyperparameters.items()},
                    "metrics": metrics,
                    "timestamp": datetime.now(),
                    "iteration": iteration,
                }
            )
        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB save failed: {e}")

    def _sync_mongodb_to_csv(self):
        try:
            all_results = list(self.collection.find().sort("iteration", 1))
            if not all_results:
                return

            headers = ",".join(["fitness", *list(self.space.keys())]) + "\n"
            with open(self.tune_csv, "w", encoding="utf-8") as f:
                f.write(headers)
                for result in all_results:
                    fitness = result["fitness"]
                    hyp_values = [result["hyperparameters"][k] for k in self.space.keys()]
                    log_row = [round(fitness, 5), *hyp_values]
                    f.write(",".join(map(str, log_row)) + "\n")

        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB to CSV sync failed: {e}")

    def _crossover(self, x: np.ndarray, alpha: float = 0.2, k: int = 9) -> np.ndarray:
        k = min(k, len(x))

        weights = x[:, 0] - x[:, 0].min() + 1e-6
        if not np.isfinite(weights).all() or weights.sum() == 0:
            weights = np.ones_like(weights)
        idxs = random.choices(range(len(x)), weights=weights, k=k)
        parents_mat = np.stack([x[i][1:] for i in idxs], 0)
        lo, hi = parents_mat.min(0), parents_mat.max(0)
        span = hi - lo
        return np.random.uniform(lo - alpha * span, hi + alpha * span)

    def _mutate(
        self,
        n: int = 9,
        mutation: float = 0.5,
        sigma: float = 0.2,
    ) -> dict[str, float]:
        x = None

        if self.mongodb:
            if results := self._get_mongodb_results(n):

                x = np.array([[r["fitness"]] + [r["hyperparameters"][k] for k in self.space.keys()] for r in results])
            elif self.collection.name in self.collection.database.list_collection_names():
                x = np.array([[0.0] + [getattr(self.args, k) for k in self.space.keys()]])

        if x is None and self.tune_csv.exists():
            csv_data = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            if len(csv_data) > 0:
                fitness = csv_data[:, 0]
                order = np.argsort(-fitness)
                x = csv_data[order][:n]

        if x is not None:
            np.random.seed(int(time.time()))
            ng = len(self.space)

            genes = self._crossover(x)

            gains = np.array([v[2] if len(v) == 3 else 1.0 for v in self.space.values()])
            factors = np.ones(ng)
            while np.all(factors == 1):
                mask = np.random.random(ng) < mutation
                step = np.random.randn(ng) * (sigma * gains)
                factors = np.where(mask, np.exp(step), 1.0).clip(0.25, 4.0)
            hyp = {k: float(genes[i] * factors[i]) for i, k in enumerate(self.space.keys())}
        else:
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}

        for k, bounds in self.space.items():
            hyp[k] = round(min(max(hyp[k], bounds[0]), bounds[1]), 5)

        if "close_mosaic" in hyp:
            hyp["close_mosaic"] = round(hyp["close_mosaic"])

        return hyp

    def __call__(self, model=None, iterations: int = 10, cleanup: bool = True):
        t0 = time.time()
        best_save_dir, best_metrics = None, None
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)

        if self.mongodb:
            self._sync_mongodb_to_csv()

        start = 0
        if self.tune_csv.exists():
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            start = x.shape[0]
            LOGGER.info(f"{self.prefix}Resuming tuning run {self.tune_dir} from iteration {start + 1}...")
        for i in range(start, iterations):

            frac = min(i / 300.0, 1.0)
            sigma_i = 0.2 - 0.1 * frac

            mutated_hyp = self._mutate(sigma=sigma_i)
            LOGGER.info(f"{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}")

            metrics = {}
            train_args = {**vars(self.args), **mutated_hyp}
            save_dir = get_save_dir(get_cfg(train_args))
            weights_dir = save_dir / "weights"
            try:

                launch = [__import__("sys").executable, "-m", "hyperimagedetect.cfg.__init__"]
                cmd = [*launch, "train", *(f"{k}={v}" for k, v in train_args.items())]
                return_code = subprocess.run(cmd, check=True).returncode
                ckpt_file = weights_dir / ("best.pt" if (weights_dir / "best.pt").exists() else "last.pt")
                metrics = torch_load(ckpt_file)["train_metrics"]
                assert return_code == 0, "training failed"

                time.sleep(1)
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                LOGGER.error(f"training failure for hyperparameter tuning iteration {i + 1}\n{e}")

            fitness = metrics.get("fitness", 0.0)
            if self.mongodb:
                self._save_to_mongodb(fitness, mutated_hyp, metrics, i + 1)
                self._sync_mongodb_to_csv()
                total_mongo_iterations = self.collection.count_documents({})
                if total_mongo_iterations >= iterations:
                    LOGGER.info(
                        f"{self.prefix}Target iterations ({iterations}) reached in MongoDB ({total_mongo_iterations}). Stopping."
                    )
                    break
            else:

                log_row = [round(fitness, 5)] + [mutated_hyp[k] for k in self.space.keys()]
                headers = "" if self.tune_csv.exists() else (",".join(["fitness", *list(self.space.keys())]) + "\n")
                with open(self.tune_csv, "a", encoding="utf-8") as f:
                    f.write(headers + ",".join(map(str, log_row)) + "\n")

            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]
            best_idx = fitness.argmax()
            best_is_current = best_idx == i
            if best_is_current:
                best_save_dir = str(save_dir)
                best_metrics = {k: round(v, 5) for k, v in metrics.items()}
                for ckpt in weights_dir.glob("*.pt"):
                    shutil.copy2(ckpt, self.tune_dir / "weights")
            elif cleanup and best_save_dir:
                shutil.rmtree(best_save_dir, ignore_errors=True)

            plot_tune_results(str(self.tune_csv))

            header = (
                f"{self.prefix}{i + 1}/{iterations} iterations complete ✅ ({time.time() - t0:.2f}s)\n"
                f"{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}\n"
                f"{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}\n"
                f"{self.prefix}Best fitness metrics are {best_metrics}\n"
                f"{self.prefix}Best fitness model is {best_save_dir}"
            )
            LOGGER.info("\n" + header)
            data = {k: float(x[best_idx, i + 1]) for i, k in enumerate(self.space.keys())}
            YAML.save(
                self.tune_dir / "best_hyperparameters.yaml",
                data=data,
                header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
            )
            YAML.print(self.tune_dir / "best_hyperparameters.yaml")