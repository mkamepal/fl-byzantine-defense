"""Post-run detection reports; attack ground truth never enters defense inputs."""

import json
import logging
from pathlib import Path

from fl_defense.detection.parameters import client_number
from fl_defense.utils.weight_archive import _publish, save_json


DEFENSES = ("clustering", "direction", "magnitude", "validation")
TITLES = {"clustering": "Clustering", "direction": "Direction",
          "magnitude": "Magnitude", "validation": "Trusted validation"}


def _metrics(matrix):
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    return {"matrix": matrix, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "observations": total, "accuracy": (tn + tp) / total if total else None,
            "malicious_precision": tp / (tp + fp) if tp + fp else None,
            "malicious_recall": tp / (tp + fn) if tp + fn else None,
            "malicious_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def build_reports(completed_rounds, assignments, num_clients):
    """Pool actual client-round observations, including rejected clients and head."""
    if type(num_clients) is not int or num_clients < 1:
        raise ValueError("Confusion matrices require a positive client population")
    known = {f"client_{i}" for i in range(num_clients)}
    if not isinstance(assignments, dict) or not set(assignments) <= known:
        raise ValueError("Attack assignments contain unknown client IDs")
    if any(attack not in ("label_flip", "backdoor", "model_replacement")
           for attack in assignments.values()):
        raise ValueError("Unknown attack assignment")
    if not completed_rounds:
        raise ValueError("No completed defense rounds for confusion matrices")
    reports = {name: {"defense": name, "per_round": {}, "decisions": []} for name in DEFENSES}
    totals = {name: [[0, 0], [0, 0]] for name in DEFENSES}
    for rnd, record in sorted(completed_rounds.items()):
        if type(rnd) is not int or rnd < 1 or record["round"] != rnd:
            raise ValueError("Invalid completed-round identity")
        clients = record["clients"]
        if not clients or not set(clients) <= known:
            raise ValueError("Missing or unknown completed-round clients")
        matrices = {name: [[0, 0], [0, 0]] for name in DEFENSES}
        for cid in sorted(clients, key=client_number):
            actual = int(cid in assignments)
            for name in DEFENSES:
                score = clients[cid].get(name, {}).get("score")
                if type(score) is not int or score not in (0, 1):
                    raise ValueError(f"Missing or nonbinary {name} score for round {rnd}, {cid}")
                predicted = 1 - score
                matrices[name][actual][predicted] += 1
                totals[name][actual][predicted] += 1
                reports[name]["decisions"].append({"round": rnd, "client_id": cid,
                    "assigned_attack": assignments.get(cid), "actual_malicious": bool(actual),
                    "score": score, "predicted_malicious": bool(predicted)})
        for name in DEFENSES:
            reports[name]["per_round"][str(rnd)] = _metrics(matrices[name])
    for name in DEFENSES:
        reports[name]["overall"] = _metrics(totals[name])
    return reports


def _plot(report, round_count):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(6.4, 5.6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    matrix = report["overall"]["matrix"]
    maximum = max(max(row) for row in matrix)
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, maximum))
    for row in range(2):
        for column in range(2):
            label = (("TN", "FP"), ("FN", "TP"))[row][column]
            value = matrix[row][column]
            axis.text(column, row, f"{label}\n{value}", ha="center", va="center", fontsize=18,
                      color="white" if value > maximum / 2 else "black")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Malicious"],
             yticklabels=["Benign", "Malicious"], xlabel="Predicted client class",
             ylabel="Actual client class (fixed attack assignment)",
             title=f"{TITLES[report['defense']]} — all {round_count} completed rounds\n"
                   f"{report['overall']['observations']} client-round decisions")
    figure.supxlabel("Score 0: flagged malicious • Score 1: passed as benign\n"
                       "Repeated clients across rounds; includes trusted head", fontsize=10)
    return figure


def save_defense_confusion(logger, completed_rounds, num_rounds):
    """Export only a fully completed combined-defense run, without model inference."""
    if set(completed_rounds) != set(range(1, num_rounds + 1)):
        logging.getLogger("flwr").info("Confusion matrices skipped: not all requested rounds completed")
        return None
    run_dir = Path(logger.run_dir)
    config = logger.config["run_config"]
    source = run_dir / "attacks" / "assignment_and_poisoning.json"
    if config.get("attacks-enabled", False):
        summary = json.loads(source.read_text(encoding="utf-8"))
        if summary["total_clients"] != config["num-clients"]:
            raise ValueError("Attack assignment population differs from this run")
        assignments = summary["assignments"]
        ground_truth_source = "attacks/assignment_and_poisoning.json"
    else:
        assignments = {}
        ground_truth_source = "config.json: attacks-enabled=false (all clients benign)"
    reports = build_reports(completed_rounds, assignments, config["num-clients"])
    metadata = {
        "flower_run_id": str(logger.config["flower_run_id"]),
        "experiment_id": logger.experiment_id,
        "completed_rounds": sorted(completed_rounds),
        "ground_truth_source": ground_truth_source,
        "ground_truth_policy": "Fixed configured malicious-client assignment, including inactive "
                               "model-replacement rounds and assignments with zero poisoned examples.",
        "score_policy": "0 = predicted malicious; 1 = predicted benign; malicious is the positive class.",
        "matrix_layout": "Rows actual, columns predicted; class order [benign, malicious]; [[TN, FP], [FN, TP]].",
        "observation_policy": "One successful client reply per completed round, including rejected clients "
                              "and the trusted head; missing/failed/unsampled clients excluded. "
                              "Repeated clients are not independent clients; counts are not sample-weighted.",
        "interpretation": "Component decisions observed in this combined-defense run, not separate "
                          "training experiments or CIFAR-10 image-classification confusion matrices.",
        "undefined_metrics": "null when denominator is zero",
    }
    directory = run_dir / "confusion_matrix"
    directory.mkdir(exist_ok=False)
    for name, report in reports.items():
        save_json(directory / f"{name}.json", {**metadata, **report})
        figure = _plot(report, len(completed_rounds))
        try:
            _publish(directory / f"{name}.png", lambda stream: figure.savefig(stream, format="png", dpi=160))
        finally:
            figure.clear()
    logging.getLogger("flwr").info("Saved all-round defense confusion matrices: %s", directory)
    return directory
