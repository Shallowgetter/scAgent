"""
Baseline runner for (s, S) policy search across config sets.
- Config names are loaded from configs/env_config_total.yaml.
- Results saved to baseline/results/<section>/<config>/<mode>_result.json
- Modes: static, hindsight (matching source sS_static / sS_hindsight)
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
# Ensure local ReplenishmentEnv package is importable without installation.
sys.path.insert(0, str(REPO_ROOT / "ReplenishmentEnv"))

import numpy as np  # noqa: E402
from ReplenishmentEnv import make_env  # noqa: E402


# -------------------- Config helpers --------------------

def _extract_config_values(section_items: Iterable[dict]) -> List[str]:
    values: List[str] = []
    for item in section_items:
        if isinstance(item, dict):
            values.extend(item.values())
    return values


def load_config_values(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    out: Dict[str, List[str]] = {}
    for section, items in data.items():
        out[section] = _extract_config_values(items)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -------------------- Runner --------------------

def run_mode(mode: str, env_name: str, vis_root: Path):
    vis_path = vis_root / mode
    ensure_dir(vis_path)
    if mode == "static":
        balance = sS_static(env_name, str(vis_path))
    elif mode == "hindsight":
        balance = sS_hindsight(env_name, str(vis_path))
    else:
        raise ValueError(f"Unknown mode {mode}")
    return balance


def to_serializable_balance(balance):
    try:
        return np.asarray(balance).tolist()
    except Exception:
        try:
            return float(balance)
        except Exception:
            return str(balance)


def summarize_balance(balance):
    try:
        arr = np.asarray(balance, dtype=float)
        return float(arr.sum())
    except Exception:
        try:
            return float(balance)
        except Exception:
            return None


def run_experiments(env_cfgs: Dict[str, List[str]], modes: List[str], results_root: Path):
    summaries = []
    for section, cfg_names in env_cfgs.items():
        for cfg_name in cfg_names:
            result_dir = results_root / section / cfg_name
            ensure_dir(result_dir)
            for mode in modes:
                start = time.time()
                status = "ok"
                balance = None
                error = None
                try:
                    balance = run_mode(mode, cfg_name, result_dir)
                except Exception as exc:  # pragma: no cover
                    status = "error"
                    error = str(exc)
                duration = time.time() - start
                record = {
                    "section": section,
                    "env_config": cfg_name,
                    "mode": mode,
                    "status": status,
                    "duration_sec": duration,
                    "balance_raw": to_serializable_balance(balance),
                    "balance_sum": summarize_balance(balance),
                    "error": error,
                    "vis_path": str(result_dir / mode),
                }
                summaries.append(record)
                out_file = result_dir / f"{mode}_result.json"
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)
                print(f"[{status}] {section}/{cfg_name}/{mode} -> {out_file} (time={duration:.2f}s)")
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Run (s,S) search baseline over multiple env configs.")
    parser.add_argument("--config-list", default="configs/env_config_total.yaml", help="YAML mapping of sections to env config names.")
    parser.add_argument("--sections", nargs="*", help="Optional subset of sections to run (e.g., ScalingUp Competition).")
    parser.add_argument("--modes", nargs="*", default=["static", "hindsight"], help="Modes to run: static, hindsight.")
    parser.add_argument("--results-dir", default="baseline/results", help="Root directory to store results.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = (REPO_ROOT / args.config_list).resolve()
    env_cfgs = load_config_values(cfg_path)
    if args.sections:
        env_cfgs = {k: v for k, v in env_cfgs.items() if k in args.sections}
    results_root = (REPO_ROOT / args.results_dir).resolve()
    ensure_dir(results_root)
    summaries = run_experiments(env_cfgs, args.modes, results_root)
    summary_path = results_root / "summary_sS.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved aggregated summary to {summary_path}")


# -------------------- (s,S) implementation (local copy) --------------------

def sS_policy(env, S, s):
    env.reset()
    done = False
    sku_count = len(env.get_sku_list())
    total_reward = np.zeros((env.warehouse_count, sku_count))
    while not done:
        mean_demand = env.get_demand_mean()
        action = (env.get_in_stock() + env.get_in_transit()) / (mean_demand + 0.0001)
        action = np.where(action < s, S - action, 0)
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward, info["balance"]


def search_sS(env, search_range=np.arange(0.0, 12.1, 1)):
    env.reset()
    sku_count = len(env.get_sku_list())
    max_reward = np.ones((sku_count)) * (-np.inf)
    best_S = np.zeros((sku_count))
    best_s = np.zeros((sku_count))

    for S in search_range:
        for s in np.arange(0, S + 0.1, 0.5):
            reward, _ = sS_policy(env, [[S] * sku_count] * env.warehouse_count, [[s] * sku_count] * env.warehouse_count)
            reward = sum(reward)
            best_S = np.where(reward > max_reward, S, best_S)
            best_s = np.where(reward > max_reward, s, best_s)
            max_reward = np.where(reward > max_reward, reward, max_reward)
    return np.ones((env.warehouse_count, sku_count)) * best_S, np.ones((env.warehouse_count, sku_count)) * best_s


def sS_hindsight(env_name, vis_path):
    env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test")
    best_S, best_s = search_sS(env_train)
    env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
    _, balance = sS_policy(env_test, best_S, best_s)
    env_test.render()
    return balance


def sS_static(env_name, vis_path):
    env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode="train")
    best_S, best_s = search_sS(env_train)
    env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
    _, balance = sS_policy(env_test, best_S, best_s)
    env_test.render()
    return balance


if __name__ == "__main__":
    main()
