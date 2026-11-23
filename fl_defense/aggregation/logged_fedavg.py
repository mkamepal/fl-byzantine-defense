"""FedAvg lifecycle logging, with optional combined defense before aggregation."""

from flwr.serverapp.strategy import FedAvg
from math import prod
import logging
from flwr.app import ConfigRecord

from fl_defense.data.client_identity import client_id
from fl_defense.detection.clustering_output import persist_clustering
from fl_defense.utils.weight_archive import save_json
from fl_defense.attacks.evaluation import log_backdoor_evaluation
from fl_defense.attacks.model_replacement import log_round_diagnostics


class LoggedFedAvg(FedAvg):
    def __init__(self, logger, num_clients, weight_archive=None, clustering_defense=None, poisoning_dir=None,
                 combined_defense=None, **kwargs):
        # A one-client baseline run must not wait forever for a second node.
        for key in ("min_train_nodes", "min_evaluate_nodes", "min_available_nodes"):
            kwargs.setdefault(key, min(2, num_clients))
        super().__init__(**kwargs)
        self.logger = logger
        self.num_clients = num_clients
        self.weight_archive = weight_archive
        self.clustering_defense = clustering_defense
        self.poisoning_dir = poisoning_dir
        self.combined_defense = combined_defense
        self.completed_defense_rounds = {}
        self._pending_defense = None
        if combined_defense is not None:
            if clustering_defense is None:
                raise ValueError("Combined trust requires an operational clustering defense")
            from fl_defense.utils.defense_excel import DefenseExcel
            from fl_defense.attacks.config import AttackConfig
            self.defense_excel = DefenseExcel(logger.run_dir, {
                "Run ID": logger.config["flower_run_id"],
                "Experiment ID": logger.experiment_id,
                "Dataset": combined_defense.evaluator.dataset_identifier, "Model": logger.config["model_name"],
                "Seed (partition)": logger.config["partition_seed"],
                "Model random seed": logger.config["model_random_seed"],
                "Trusted head ID": clustering_defense.config.head_client,
                "Validation-data source": combined_defense.evaluator.source,
                **combined_defense.config.effective(),
                "Clustering settings": clustering_defense.config.effective(),
                "Effective attack settings": AttackConfig.from_config(logger.config["run_config"]).effective(),
                "Experiment configuration (including attacks)": logger.config["run_config"],
                "Code revision": logger.config.get("code_revision"),
            })
        if clustering_defense is not None:
            from fl_defense.detection.parameters import client_number
            if client_number(clustering_defense.config.head_client) >= num_clients:
                raise ValueError("Trusted reference is outside configured client partitions")

    def start(self, grid, initial_arrays, num_rounds=3, timeout=3600, train_config=None,
              evaluate_config=None, evaluate_fn=None, should_stop=None):
        """Wrap Flower's final evaluation callback, after both evaluation stages.

        This does not add a second training loop. Incomplete rounds never reach
        the workbook, including failures in optional centralized evaluation.
        """
        def completed(server_round, arrays):
            metrics = evaluate_fn(server_round, arrays) if evaluate_fn is not None else None
            if server_round > 0 and self.combined_defense is not None:
                if self._pending_defense is None or self._pending_defense["round"] != server_round:
                    raise ValueError("Missing defense results for completed round")
                record = self._pending_defense
                self.completed_defense_rounds[server_round] = record
                directory = self.logger.run_dir / "defense"
                directory.mkdir(exist_ok=True)
                save_json(directory / f"round_{server_round:03d}.json", record)
                self.defense_excel.save(self.completed_defense_rounds)
                self._pending_defense = None
            return metrics

        extra = {"should_stop": should_stop} if should_stop is not None else {}
        result = super().start(grid=grid, initial_arrays=initial_arrays, num_rounds=num_rounds, timeout=timeout,
                             train_config=train_config, evaluate_config=evaluate_config,
                             evaluate_fn=completed if self.combined_defense is not None else evaluate_fn, **extra)
        if self.combined_defense is not None:
            from fl_defense.utils.defense_confusion import save_defense_confusion
            save_defense_confusion(self.logger, self.completed_defense_rounds, num_rounds)
        return result

    def configure_train(self, server_round, arrays, config, grid):
        if self._pending_defense is not None:
            raise ValueError("Previous defense round has not completed evaluation/export")
        config = ConfigRecord(dict(config))
        config['worker-memory-dir'] = str((self.logger.run_dir / 'worker_memory').resolve())
        if self.poisoning_dir is not None:
            config['poisoning-data-dir'] = self.poisoning_dir
        messages = list(super().configure_train(server_round, arrays, config, grid))
        self._training_nodes = {m.metadata.dst_node_id for m in messages}
        if self.clustering_defense is not None:
            dimension = sum(prod(p.shape) for p in self.clustering_defense.schema if p.trainable)
            try:
                self.clustering_defense.config.validate_population(len(messages), dimension)
            except ValueError as error:
                self._record_clustering_failure(server_round, error)
                raise
            self.clustering_defense.begin_round(server_round, arrays.to_numpy_ndarrays())
        if self.weight_archive is not None:
            self.weight_archive.begin_round(server_round, arrays, [m.metadata.dst_node_id for m in messages])
        self.logger.server_event(server_round, {
            "round": server_round,
            "participating_clients": len(messages),
        })
        self.logger.flush()
        return messages

    def configure_evaluate(self, server_round, arrays, config, grid):
        config = ConfigRecord(dict(config))
        config['worker-memory-dir'] = str((self.logger.run_dir / 'worker_memory').resolve())
        messages = list(super().configure_evaluate(server_round, arrays, config, grid))
        self.logger.server_event(server_round, {"evaluation_clients": len(messages)})
        self.logger.flush()
        return messages

    def _log_replies(self, replies, server_round, training):
        category = "training" if training else "validation"
        count = 0
        for message in replies:
            if message.has_error():
                self.logger.text(
                    f"server/{category}_errors/node_{message.metadata.src_node_id}",
                    message.error.reason, server_round,
                )
                continue
            count += 1
            info = message.content["client-info"]
            partition_id = info["partition-id"]
            cid = client_id(partition_id)
            if not 0 <= partition_id < self.num_clients or info["client-id"] != cid:
                raise ValueError("Reply client identity does not match a known partition")
            values = message.content["metrics"]
            mapping = {"train_loss": "loss", "train_acc": "accuracy"} if training else {"eval_loss": "loss", "eval_acc": "accuracy"}
            scalars = {tag: values[key] for key, tag in mapping.items() if key in values}
            scalars["round"] = server_round
            scalars["num_samples"] = values["num-examples"]
            if training:
                scalars["local_epochs"] = info["local-epochs"]
            self.logger.client_event(cid, category, server_round, scalars)
            self.logger.text(
                f"clients/{cid}/identity",
                f"Client: {cid} | Partition: {partition_id} | Role: {info['role']} | Flower node: {message.metadata.src_node_id}",
                server_round,
            )
        return count

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        if self.weight_archive is not None:
            try:
                clients = self.weight_archive.capture_clients(server_round, replies, self.arrayrecord_key)
            except Exception as error:
                if self.clustering_defense is not None:
                    self._record_clustering_failure(server_round, error)
                    self.clustering_defense.release()
                    self.weight_archive.baseline = None
                raise
            for client in clients:
                shift = client["shift"]
                self.logger.client_event(client["client_id"], "weight_shift", server_round,
                                         {key: shift[key] for key in ("l2", "max_abs", "relative_l2", "trainable_l2", "nonfinite_elements")
                                          if shift[key] is not None})
        selected_replies = replies
        if self.clustering_defense is not None:
            selected_replies = self._filter_clustered_replies(server_round, replies)
        result = super().aggregate_train(server_round, selected_replies)
        if self.combined_defense is not None and result[0] is None:
            self._pending_defense = None
            raise ValueError("Combined-defense round did not produce an aggregated model")
        if self.combined_defense is not None:
            self._log_trust_selection(server_round)
        if self.weight_archive is not None:
            self.weight_archive.finish_round(server_round, result[0])
        if result[0] is not None:
            log_round_diagnostics(self.logger, self.poisoning_dir, server_round, replies, selected_replies)
        count = self._log_replies(replies, server_round, training=True)
        self.logger.server_event(server_round, {"successful_updates": count,
                                                "training_error_replies": sum(m.has_error() for m in replies)})
        if result[1] is not None:
            self.logger.server_event(server_round, dict(result[1]), category="server/aggregated_training")
        self.logger.flush()
        return result

    def _log_trust_selection(self, server_round):
        """Report the same successful client decisions used to filter FedAvg."""
        from fl_defense.detection.parameters import client_number

        records = self._pending_defense["clients"]
        accepted, rejected = [], []
        for cid in sorted(records, key=client_number):
            value = records[cid]
            entry = f"{cid} (trust={value['trust_score']:.6g})"
            (accepted if value["selected"] else rejected).append(entry)
        threshold = self.combined_defense.config.trust_threshold
        lines = [f"Round {server_round} global aggregation (successful client updates; trust >= {threshold:.6g})",
                 f"Accepted clients ({len(accepted)}): {', '.join(accepted) or 'none'}",
                 f"Rejected clients ({len(rejected)}): {', '.join(rejected) or 'none'}"]
        logging.getLogger("flwr").info("\n%s", "\n".join(lines))

    def _filter_clustered_replies(self, server_round, replies):
        """Abort on invalid input; never replace failure with accepting all clients."""
        defense = self.clustering_defense
        try:
            clients, messages, seen_nodes = {}, {}, set()
            for message in replies:
                node = message.metadata.src_node_id
                if node not in self._training_nodes or node in seen_nodes:
                    raise ValueError("Unrequested or duplicate training node reply")
                seen_nodes.add(node)
                if message.has_error():
                    continue
                info = message.content["client-info"]
                partition = info["partition-id"]
                cid = client_id(partition)
                if not 0 <= partition < self.num_clients or info["client-id"] != cid or cid in clients:
                    raise ValueError("Invalid or duplicate client identity in clustering")
                if info.get("weight-convention") != "full-model" or info.get("server-round") != server_round:
                    raise ValueError("Clustering requires full models from this round")
                arrays = message.content[self.arrayrecord_key]
                if list(arrays) != [str(i) for i in range(len(defense.schema))]:
                    raise ValueError("Model array keys/order mismatch")
                clients[cid] = (arrays.to_numpy_ndarrays(), info["weight-schema"])
                messages[cid] = message
            result, artifacts = defense.analyze(server_round, clients)
            persist_clustering(self.logger, result, artifacts)
            if self.combined_defense is not None:
                record = self.combined_defense.analyze(clients, result, defense.reference, defense.schema)
                selected = [cid for cid, value in record["clients"].items() if value["selected"]]
                for cid, value in record["clients"].items():
                    self.logger.client_event(cid, "detection/trust", server_round, {
                        "score": value["trust_score"], "selected": int(value["selected"]),
                        **{name + "_score": value[name]["score"] for name in
                           ("clustering", "direction", "magnitude", "validation")},
                    })
                self.logger.server_event(server_round, {"selected_clients": len(selected),
                                                       "rejected_clients": len(clients) - len(selected)},
                                         category="detection/trust")
                self._pending_defense = record
                return [messages[cid] for cid in selected]
            return [messages[cid] for i, cid in enumerate(result.client_ids) if result.clustering_scores[i] == 1]
        except Exception as error:
            self._record_clustering_failure(server_round, error)
            if self.weight_archive is not None:
                self.weight_archive.baseline = None
            raise
        finally:
            defense.release()

    def _record_clustering_failure(self, server_round, error):
        directory = self.logger.run_dir / "clustering"
        directory.mkdir(exist_ok=True)
        save_json(directory / f"round_{server_round:03d}_failure.json", {
            "round_id": server_round, "status": "aborted", "reason": str(error),
            "effective_config": self.clustering_defense.config.effective(),
            "combined_config": self.combined_defense.config.effective() if self.combined_defense is not None else None,
        })
        self.logger.text("detection/clustering/failure", str(error), server_round)
        self.logger.flush()

    def aggregate_evaluate(self, server_round, replies):
        replies = list(replies)
        result = super().aggregate_evaluate(server_round, replies)
        count = self._log_replies(replies, server_round, training=False)
        log_backdoor_evaluation(self.logger, server_round, replies)
        self.logger.server_event(server_round, {"successful_evaluations": count,
                                                "evaluation_error_replies": sum(m.has_error() for m in replies)})
        if result is not None:
            metrics = {tag: result[key] for key, tag in
                       (("eval_acc", "global_accuracy"), ("eval_loss", "global_loss")) if key in result}
            self.logger.server_event(server_round, metrics)
        self.logger.flush()
        return result
