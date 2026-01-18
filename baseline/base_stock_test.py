"""
Baseline runner for Base-Stock algorithm (static + dynamic) across config sets.

- Config names are loaded from configs/env_config_total.yaml.
- Each top-level section (ScalingUp, Competition, Robustness, ...) is written to
  baseline/results/<section>/<config_name>/result.json
- Uses ReplenishmentEnv.Baseline.OR_algorithm.base_stock.{BS_static, BS_dynamic}
- run all sections, both modes: python -m baseline.base_stock_test
- subset: python -m baseline.base_stock_test --sections ScalingUp Competition --modes static dynamic 
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

# Local copy of base-stock logic to avoid cross-import issues.
import numpy as np  # noqa: E402
import cvxpy as cp  # noqa: E402
from ReplenishmentEnv import make_env  # noqa: E402


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


def run_mode(mode: str, env_name: str, vis_root: Path):
    vis_path = vis_root / mode
    ensure_dir(vis_path)
    if mode == "static":
        balance, metrics = BS_static(env_name, str(vis_path))
    elif mode == "dynamic":
        balance, metrics = BS_dynamic(env_name, str(vis_path))
    else:
        raise ValueError(f"Unknown mode {mode}")
    return balance, metrics


def to_serializable_balance(balance):
    try:
        import numpy as np

        return np.asarray(balance).tolist()
    except Exception:
        try:
            return float(balance)
        except Exception:
            return str(balance)


def summarize_balance(balance):
    import numpy as np

    try:
        arr = np.asarray(balance, dtype=float)
        return float(arr.sum())
    except Exception:
        try:
            return float(balance)
        except Exception:
            return None


def extract_metrics(info):
    # Extract metrics from info dict produced by MetricsWrapper
    bwe = info.get("metrics", {}).get("bwe", {}) if isinstance(info, dict) else {}
    tau = info.get("metrics", {}).get("tau_rec", {}) if isinstance(info, dict) else {}
    # Stability: use max of per_tier tau_rec (higher = slower recovery)
    tau_per_tier = tau.get("per_tier") or []
    stability_max = max([t for t in tau_per_tier if t is not None], default=None)
    return {
        "bwe_mean": bwe.get("mean"),
        "bwe_per_tier": bwe.get("per_tier"),
        "tau_rec_mean": tau.get("mean"),
        "tau_rec_per_tier": tau_per_tier,
        "stability_max_tau": stability_max,
    }


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
                metrics_info = None
                try:
                    balance, info_metrics = run_mode(mode, cfg_name, result_dir)
                    metrics_info = info_metrics
                except Exception as exc:  # pragma: no cover - runtime safety
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
                    "metrics": metrics_info,
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
    parser = argparse.ArgumentParser(description="Run Base-Stock baseline over multiple env configs.")
    parser.add_argument("--config-list", default="configs/env_config_total.yaml", help="YAML mapping of sections to env config names.")
    parser.add_argument("--sections", nargs="*", help="Optional subset of sections to run (e.g., ScalingUp Competition).")
    parser.add_argument("--modes", nargs="*", default=["static", "dynamic"], help="Modes to run: static, dynamic.")
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
    summary_path = results_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved aggregated summary to {summary_path}")


# -------------------- Base-stock implementation (local copy) --------------------


def get_multilevel_stock_level(env):
    stock_levels = np.zeros((len(env.get_warehouse_list()), len(env.get_sku_list())))
    for sku_index, sku in enumerate(env.get_sku_list()):
        selling_price = np.array(env.get_selling_price(sku=sku))
        procurement_cost = np.array(env.get_procurement_cost(sku=sku))
        demand = np.array(env.get_demand(sku=sku))
        average_vlt = env.get_average_vlt(sku=sku).max()
        holding_cost = np.array(env.get_holding_cost(sku=sku))
        if demand.ndim == 2:
            demand = demand[-1].reshape(-1)
        stock_level = get_multilevel_single_stock_level(
            selling_price,
            procurement_cost,
            demand,
            average_vlt,
            holding_cost,
            env.warehouse_count,
        ).reshape(-1,)
        stock_levels[:, sku_index] = stock_level
    return stock_levels


def get_multilevel_single_stock_level(
    selling_price: np.array,
    procurement_cost: np.array,
    demand: np.array,
    vlt: int,
    holding_cost: np.array,
    warehouses: int,
) -> np.ndarray:
    time_hrz_len = len(selling_price[0])
    stocks = cp.Variable((warehouses, time_hrz_len + 1), integer=True)
    transits = cp.Variable((warehouses, time_hrz_len + 1), integer=True)
    sales = cp.Variable((warehouses, time_hrz_len), integer=True)
    buy = cp.Variable((warehouses, time_hrz_len + vlt), integer=True)
    buy_in = cp.Variable((warehouses, time_hrz_len), integer=True)
    buy_arv = cp.Variable((warehouses, time_hrz_len), integer=True)
    stock_level = cp.Variable((warehouses, 1), integer=True)
    profit = cp.Variable(1)
    common_constraints = [
        stocks >= 0,
        transits >= 0,
        sales >= 0,
        buy >= 0,
    ]
    intralevel_constraints = [
        stocks[:, 1: time_hrz_len + 1] == stocks[:, 0:time_hrz_len] + buy_arv - sales,
        transits[:, 1: time_hrz_len + 1] == transits[:, 0:time_hrz_len] - buy_arv + buy_in,
        sales <= stocks[:, 0:time_hrz_len],
        buy_in == buy[:, vlt: time_hrz_len + vlt],
        buy_arv == buy[:, 0:time_hrz_len],
        stock_level == stocks[:, 0:time_hrz_len] + transits[:, 0:time_hrz_len] + buy_in,
        transits[:, 0] == cp.sum(buy[:, :vlt], axis=1),
    ]
    intralevel_constraints.append(
        profit
        == cp.sum(
            cp.multiply(selling_price, sales)
            - cp.multiply(procurement_cost, buy_in)
            - cp.multiply(holding_cost, stocks[:, 1:]),
        )
        - cp.sum(cp.multiply(procurement_cost[:, 0], transits[:, 0]))
        - cp.sum(cp.multiply(procurement_cost[:, 0], stocks[:, 0]))
    )
    interlevel_constraints = []
    for i in range(warehouses):
        if i != warehouses - 1:
            interlevel_constraints.append(sales[i] == buy_in[i + 1])
        else:
            interlevel_constraints.append(sales[i] <= demand)

    constraints = common_constraints + intralevel_constraints + interlevel_constraints
    obj = cp.Maximize(profit)
    prob = cp.Problem(obj, constraints)

    prob.solve(verbose=False)
    if prob.status != "optimal":
        prob.solve(solver=cp.GLPK_MI, verbose=False, max_iters=1000)
    if prob.status != "optimal":
        assert prob.status == "optimal", "can't find optimal solution for SKU stock level"
    return stock_level.value


def multilevel_base_stock(env, update_freq=7, static_stock_levels=None):
    env.reset()
    current_step = 0
    is_done = False
    sku_count = len(env.get_sku_list())
    total_reward = np.zeros((env.warehouse_count, sku_count))
    stock_level_list = [[] for _ in range(len(env.get_warehouse_list()))]
    last_info = {}
    while not is_done:
        if current_step % update_freq == 0:
            if isinstance(static_stock_levels, np.ndarray):
                stock_levels = static_stock_levels
            else:
                stock_levels = get_multilevel_stock_level(env)
        for i in range(len(env.get_warehouse_list())):
            stock_level_list[i].append(stock_levels[i])

        replenish = stock_levels - env.get_in_stock() - env.get_in_transit()
        replenish = np.where(replenish >= 0, replenish, 0) / (env.get_demand_mean() + 0.00001)
        states, reward, is_done, info = env.step(replenish)
        last_info = info or {}
        total_reward += reward
        current_step += 1

    return info["balance"], last_info


def BS_static(env_name, vis_path):
    """Base stock algorithm static mode."""
    env_train = make_env(env_name, wrapper_names=["OracleWrapper", "MetricsWrapper"], mode="train", vis_path=vis_path)
    env_train.reset()
    static_stock_levels = get_multilevel_stock_level(env_train)
    env_test = make_env(env_name, wrapper_names=["OracleWrapper", "MetricsWrapper"], mode="test", vis_path=vis_path)
    balance, info = multilevel_base_stock(env_test, static_stock_levels=static_stock_levels)
    env_test.render()
    return balance, extract_metrics(info)


def BS_dynamic(env_name, vis_path):
    """Base stock algorithm dynamic mode."""
    env = make_env(env_name, wrapper_names=["HistoryWrapper", "MetricsWrapper"], mode="test", vis_path=vis_path)
    balance, info = multilevel_base_stock(env)
    env.render()
    return balance, extract_metrics(info)


if __name__ == "__main__":
    main()
