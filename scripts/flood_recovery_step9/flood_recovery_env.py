
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# 把 SUMO + 洪水 + tactic library 包装成 Gymnasium 环境
# 启动 SUMO；
# 运行到洪水 onset；
# 调用 FloodEngine；
# 解码 RL 动作；
# 调用 TacticManager；
# 逐秒推进 SUMO；
# 计算 PR；
# 计算 reward；
# 判断 recovery；
# 返回 observation、reward、terminated、truncated、info。
SUMO_HOME = Path(r"D:\Eclipse\SUMO")
PROJECT_ROOT = Path(r"D:\SUMO_A9_Project")

os.environ["SUMO_HOME"] = str(SUMO_HOME)
tools_path = SUMO_HOME / "tools"
if str(tools_path) not in sys.path:
    sys.path.append(str(tools_path))

import sumolib

import traci as traci_socket

try:
    import libsumo
except ImportError:
    libsumo = None

from action_space import ActionCodec
from baseline_provider import BaselineTrajectory
from episode_logger import EpisodeLogger
from flood_engine import FloodEngine
from flood_scenario import FloodScenario, get_fixed_scenario
from impact_monitor import EpochImpact, ImpactMonitor
from network_metrics import collect_network_step_metrics
from recovery_metrics import (
    RecoveryConfig,
    RecoveryTracker,
    compute_performance_ratio,
)
from tactic_manager import TacticApplicationResult, TacticManager
from tactic_sites import get_location_ids

from segment_state_monitor import SegmentStateMonitor
from state_feature_builder import StateFeatureBuilder
from state_segments import get_state_segment_ids


class FloodRecoveryEnv(gym.Env):
    """
    Shared Step 7/Step 8 base environment.

    Step 7:
        fixed flood + PPO-compatible MultiDiscrete tactic control.

    Step 8:
        RandomizedFloodRecoveryEnv subclasses this class and replaces
        self.scenario before calling reset().

    Action:
        MultiDiscrete([5 tactics, n_locations, 4 intensities])

    Episode logging can be disabled during PPO training to avoid excessive
    per-second CSV I/O, while remaining enabled for evaluation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        scenario_name: str = "moderate",
        decision_period_s: int = 300,
        recovery_threshold: float = 0.90,
        stability_window_s: int = 600,
        sumo_seed: int = 42,
        gui: bool = False,
        sumo_backend: str = "traci",
        output_dir: Path | None = None,
        pr_progress_weight: float = 1.0, #step9
        switching_penalty: float = 0.01,
        queue_growth_weight: float = 0.001,

        terminal_bonus: float = 1.0,
        terminal_failure_penalty: float = 1.0,

        enable_episode_logging: bool = True,
        policy_version: str = "part4_reward_v1",
    ):
        super().__init__()

        self.scenario: FloodScenario = get_fixed_scenario(
            scenario_name
        )
        self.decision_period_s = int(decision_period_s)
        self.sumo_seed = int(sumo_seed)
        self.gui = bool(gui)
        self.sumo_backend = str(
            sumo_backend
        ).lower().strip()

        if self.sumo_backend not in {
            "traci",
            "libsumo",
        }:
            raise ValueError(
                "sumo_backend must be "
                "'traci' or 'libsumo'."
            )

        if self.gui and self.sumo_backend == "libsumo":
            raise ValueError(
                "GUI debugging must use "
                "sumo_backend='traci'."
            )

        if self.sumo_backend == "libsumo":

            if libsumo is None:
                raise ImportError(
                    "libsumo is not available. "
                    "Install/use the libsumo package "
                    "for training."
                )

            self.traci = libsumo

        else:
            self.traci = traci_socket
        self.pr_progress_weight = float(
            pr_progress_weight
        )

        self.switching_penalty = float(
            switching_penalty
        )

        self.queue_growth_weight = float(
            queue_growth_weight
        )

        self.terminal_bonus = float(
            terminal_bonus
        )

        self.terminal_failure_penalty = float(
            terminal_failure_penalty
        )

        for name, value in {
            "pr_progress_weight": self.pr_progress_weight,
            "switching_penalty": self.switching_penalty,
            "queue_growth_weight": self.queue_growth_weight,
            "terminal_bonus": self.terminal_bonus,
            "terminal_failure_penalty": (
                    self.terminal_failure_penalty
            ),
        }.items():

            if value < 0.0:
                raise ValueError(
                    f"{name} must be non-negative."
                )
        self.enable_episode_logging = bool(enable_episode_logging)
        self.policy_version = str(policy_version)

        self.net_file = PROJECT_ROOT / "sumo" / "a8_corridor.net.xml"
        self.cfg_file = PROJECT_ROOT / "routes" / "a8_od_spsa.sumocfg"
        self.detector_file = (
            PROJECT_ROOT
            / "detectors"
            / "a8_detectors_merged.add.xml"
        )
        self.label_file = (
            PROJECT_ROOT / "labels" / "a8_labels.add.xml"
        )
        self.gui_settings_file = (
            PROJECT_ROOT / "labels" / "a8_gui_settings.xml"
        )
        self.baseline_csv = (
            PROJECT_ROOT
            / "heat_edges"
            / "a8_step_metrics_od_reroute_calibrated.csv"
        )
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else (
                PROJECT_ROOT
                / "validation_results"
                / "step9_ppo_env"
            )
        )

        self.recovery_config = RecoveryConfig(
            threshold=float(recovery_threshold),
            stability_window_s=int(stability_window_s),
            search_start_t=float(self.scenario.peak_end_t),
        )

        self.location_ids = get_location_ids()

        self.action_codec = ActionCodec(
            self.location_ids
        )

        self.state_builder = StateFeatureBuilder(
            location_ids=self.location_ids,
            segment_ids=get_state_segment_ids(),
        )

        self.state_builder.save_schema(
            self.output_dir / "state_schema.json"
        )

        self.action_space = spaces.MultiDiscrete(
            self.action_codec.nvec
        )

        self.observation_space = spaces.Box(
            low=np.zeros(
                self.state_builder.state_dim,
                dtype=np.float32,
            ),
            high=np.full(
                self.state_builder.state_dim,
                np.inf,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.baseline = BaselineTrajectory(self.baseline_csv)
        self.impact_monitor = ImpactMonitor()

        self.net = None
        self.flood_engine: FloodEngine | None = None
        self.tactic_manager: TacticManager | None = None
        self.segment_monitor: (
                SegmentStateMonitor | None
        ) = None
        self.recovery_tracker: RecoveryTracker | None = None
        self.episode_logger: EpisodeLogger | None = None

        self.last_pr = 1.0
        self.last_metrics: dict[str, Any] | None = None
        self.decision_epoch = 0
        self.last_segment_queue_lengths: dict[
            str,
            float,
        ] = {}
        self._sumo_running = False

    def _build_sumo_command(self) -> list[str]:
        binary_name = "sumo-gui.exe" if self.gui else "sumo.exe"
        binary = SUMO_HOME / "bin" / binary_name

        command = [
            str(binary),
            "-c",
            str(self.cfg_file),
            "-a",
            ",".join([
                str(self.detector_file),
                str(self.label_file),
            ]),
            "--start",
            "--seed",
            str(self.sumo_seed),
            "--duration-log.statistics",
        ]

        if self.gui:
            command.extend([
                "--gui-settings-file",
                str(self.gui_settings_file),
                "--delay",
                "10",
            ])

        return command

    def _validate_required_files(self) -> None:
        required = [
            self.net_file,
            self.cfg_file,
            self.detector_file,
            self.label_file,
            self.baseline_csv,
        ]
        if self.gui:
            required.append(self.gui_settings_file)

        for path in required:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {path}"
                )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._sumo_running:
            self.close()

        if seed is not None:
            self.sumo_seed = int(seed)

        self._validate_required_files()
        self.traci.start(
            self._build_sumo_command()
        )
        self._sumo_running = True

        self.net = sumolib.net.readNet(str(self.net_file))
        self.flood_engine = FloodEngine(
            traci_api=self.traci,
            net=self.net,
            scenario=self.scenario,
        )

        self.tactic_manager = TacticManager(
            traci_api=self.traci,
            net=self.net,
        )

        self.segment_monitor = SegmentStateMonitor(
            traci_api=self.traci,
            net=self.net,
            window_s=self.decision_period_s,
        )

        self.segment_monitor.reset()

        self.recovery_tracker = RecoveryTracker(
            config=self.recovery_config,
            flood_onset_t=self.scenario.onset_t,
        )
        self.impact_monitor.reset()
        self.episode_logger = (
            EpisodeLogger(
                output_dir=self.output_dir,
                scenario=self.scenario,
                policy_version=self.policy_version,
                sumo_seed=self.sumo_seed,
                decision_period_s=self.decision_period_s,
            )
            if self.enable_episode_logging
            else None
        )

        self.last_pr = 1.0
        self.decision_epoch = 0

        while (
                self.traci.simulation.getTime()
                < self.scenario.onset_t
        ):
            current_time = self.traci.simulation.getTime()

            stage = self.flood_engine.apply(
                current_time
            )

            self.traci.simulationStep()

            self.segment_monitor.collect_step()

        stage = self.flood_engine.get_stage(
            self.traci.simulation.getTime()
        )

        self.last_metrics = collect_network_step_metrics(
            traci_api=self.traci,
            flood_stage=stage.value,
        )
        initial_pr = self._performance_ratio(self.last_metrics)
        self.last_pr = initial_pr

        if self.episode_logger is not None:
            self.episode_logger.set_initial_state({
                "time_s": float(self.last_metrics["time_s"]),
                "mean_speed_mps": float(
                    self.last_metrics["mean_speed_mps"]
                ),
                "running_vehicle_count": int(
                    self.last_metrics["running_vehicle_count"]
                ),
                "mean_waiting_time_s": float(
                    self.last_metrics["mean_waiting_time_s"]
                ),
                "mean_time_loss_s": float(
                    self.last_metrics["mean_time_loss_s"]
                ),
                "halting_vehicle_count": int(
                    self.last_metrics["halting_vehicle_count"]
                ),
                "pr_t": initial_pr,
            })

        initial_state_vector = self._get_state(
            self.last_metrics,
            initial_pr,
        )

        self.last_segment_queue_lengths = (
            self._get_segment_queue_lengths()
        )

        if self.episode_logger is not None:
            self.episode_logger.log_state(
                time_s=float(
                    self.last_metrics["time_s"]
                ),
                decision_epoch=0,
                feature_names=(
                    self.state_builder.feature_names
                ),
                state_vector=initial_state_vector,
            )

        return initial_state_vector, {
            "scenario_id": self.scenario.scenario_id,
            "time_s": float(
                self.traci.simulation.getTime()
            ),
            "pr_t": initial_pr,
            "action_space_nvec": (
                self.action_codec.nvec.tolist()
            ),
            "state_dim": self.state_builder.state_dim,
            "state_schema_version": (
                "part3_state_v1"
            ),
        }

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid MultiDiscrete action: {action}")
        if self.tactic_manager is None:
            raise RuntimeError("Call reset() before step().")

        decoded = self.action_codec.decode(action)

        update_result = self.tactic_manager.set_action(
            decoded
        )

        management_changed = update_result.changed

        penalize_change = (
                self.decision_epoch > 0
                and management_changed
        )

        management_state = (
            self.tactic_manager.active_measures_snapshot()
        )

        self.decision_epoch += 1

        epoch_impact = EpochImpact()
        metrics = None
        recovery_status = self.recovery_tracker.status
        aggregate_result = TacticApplicationResult(action=decoded)

        for _ in range(self.decision_period_s):
            current_time = self.traci.simulation.getTime()

            # 1. Remove physical effects left by previous
            #    management layer, but keep persistent definitions.
            self.tactic_manager.restore_management_base()

            # 2. Reconstruct physical flood state.
            stage = self.flood_engine.apply(
                current_time
            )

            # 3. Layer ALL persistent management measures.
            results = self.tactic_manager.apply_active(
                current_time
            )

            second_changed_lanes = 0
            second_changed_edges = 0
            second_rerouted = 0

            for result in results:
                second_changed_lanes += (
                    result.changed_lanes
                )
                second_changed_edges += (
                    result.changed_edges
                )
                second_rerouted += (
                    result.rerouted_vehicles
                )

                if result.invalid_reason is not None:
                    aggregate_result.invalid_reason = (
                        result.invalid_reason
                    )

            aggregate_result.changed_lanes = max(
                aggregate_result.changed_lanes,
                second_changed_lanes,
            )

            aggregate_result.changed_edges = max(
                aggregate_result.changed_edges,
                second_changed_edges,
            )

            aggregate_result.rerouted_vehicles += (
                second_rerouted
            )

            self.traci.simulationStep()

            self.segment_monitor.collect_step()

            step_impact = (
                self.impact_monitor.collect_step(self.traci)
            )
            epoch_impact.co2_mg += step_impact.co2_mg
            epoch_impact.hard_braking_events += (
                step_impact.hard_braking_events
            )
            epoch_impact.queue_exposure_veh_s += (
                step_impact.queue_exposure_veh_s
            )
            epoch_impact.time_loss_delta_s += (
                step_impact.time_loss_delta_s
            )

            metrics = collect_network_step_metrics(
                traci_api=self.traci,
                flood_stage=stage.value,
            )
            pr_second = self._performance_ratio(metrics)
            recovery_status = self.recovery_tracker.update(
                current_time_s=float(metrics["time_s"]),
                pr_t=pr_second,
            )

            if self.episode_logger is not None:
                self.episode_logger.log_second({
                    "time_s": float(metrics["time_s"]),
                    "decision_epoch": self.decision_epoch,
                    "tactic_name": decoded.tactic_name,
                    "location_id": decoded.location_id,
                    "intensity_name": decoded.intensity_name,
                    "flood_stage": metrics["flood_stage"],
                    "running_vehicle_count": int(
                        metrics["running_vehicle_count"]
                    ),
                    "mean_speed_mps": float(
                        metrics["mean_speed_mps"]
                    ),
                    "baseline_mean_speed_mps": (
                        self.baseline.get_mean_speed(
                            float(metrics["time_s"])
                        )
                    ),
                    "pr_t": pr_second,
                    "co2_step_mg": step_impact.co2_mg,
                    "hard_braking_events_step": (
                        step_impact.hard_braking_events
                    ),
                    "queue_exposure_step_veh_s": (
                        step_impact.queue_exposure_veh_s
                    ),
                    "time_loss_step_s": (
                        step_impact.time_loss_delta_s
                    ),
                    "recovered": recovery_status.recovered,
                })

            if recovery_status.recovered:
                break
            if (
                self.traci.simulation.getTime()
                >= self.scenario.max_sim_time_t
            ):
                break

        assert metrics is not None

        pr_t = self._performance_ratio(metrics)
        prior_pr = self.last_pr
        delta_pr = float(pr_t - prior_pr)
        current_queues = (self._get_segment_queue_lengths())
        queue_growth = self._compute_queue_growth(current_queues)

        self.last_pr = pr_t
        self.last_metrics = metrics
        self.last_segment_queue_lengths = current_queues

        terminated = bool(recovery_status.recovered)
        truncated = bool(
            float(metrics["time_s"])
            >= self.scenario.max_sim_time_t
            and not terminated
        )
        terminal_reward = 0.0

        if terminated:

            recovery_time_s = float(
                recovery_status.recovery_time_s
            )

            recovery_horizon_s = max(
                1.0,
                float(
                    self.scenario.max_sim_time_t
                    - self.scenario.onset_t
                ),
            )

            speed_factor = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - recovery_time_s
                    / recovery_horizon_s,
                ),
            )

            terminal_reward = (
                    self.terminal_bonus
                    * speed_factor
            )

        elif truncated:

            terminal_reward = (
                -self.terminal_failure_penalty
            )

        pr_progress_reward = (
                self.pr_progress_weight
                * delta_pr
        )

        action_switch_cost = (
            self.switching_penalty
            if penalize_change
            else 0.0
        )

        queue_growth_penalty = (
                self.queue_growth_weight
                * queue_growth
        )

        reward = (
                pr_progress_reward
                - action_switch_cost
                - queue_growth_penalty
                + terminal_reward
        )

        impact_totals = self.impact_monitor.totals()
        action_result_dict = aggregate_result.as_dict()
        action_result_dict["operation"] = (
            update_result.operation
        )

        if self.episode_logger is not None:
            self.episode_logger.log_epoch({
                "time_s": float(metrics["time_s"]),
                "decision_epoch": self.decision_epoch,
                **action_result_dict,
                "flood_stage": metrics["flood_stage"],
                "running_vehicle_count": int(
                    metrics["running_vehicle_count"]
                ),
                "mean_speed_mps": float(metrics["mean_speed_mps"]),
                "baseline_mean_speed_mps": (
                    self.baseline.get_mean_speed(
                        float(metrics["time_s"])
                    )
                ),
                "pr_t": pr_t,
                "mean_waiting_time_s": float(
                    metrics["mean_waiting_time_s"]
                ),
                "mean_time_loss_s": float(
                    metrics["mean_time_loss_s"]
                ),
                "halting_vehicle_count": int(
                    metrics["halting_vehicle_count"]
                ),
                "delta_pr": delta_pr,

                "pr_progress_reward": (
                    pr_progress_reward
                ),

                "action_switch_cost": (
                    action_switch_cost
                ),

                "queue_growth_veh": (
                    queue_growth
                ),

                "queue_growth_penalty": (
                    queue_growth_penalty
                ),

                "terminal_reward": (
                    terminal_reward
                ),
                "reward": reward,
                "co2_epoch_kg": epoch_impact.co2_kg,
                "hard_braking_events_epoch": (
                    epoch_impact.hard_braking_events
                ),
                "queue_exposure_epoch_veh_h": (
                    epoch_impact.queue_exposure_veh_s / 3600.0
                ),
                "time_loss_epoch_veh_h": (
                    epoch_impact.time_loss_delta_veh_h
                ),
                "co2_cumulative_kg": impact_totals["co2_total_kg"],
                "hard_braking_events_cumulative": (
                    impact_totals["hard_braking_events_total"]
                ),
                "queue_exposure_cumulative_veh_h": (
                    impact_totals["queue_exposure_veh_h"]
                ),
                "time_loss_cumulative_veh_h": (
                    impact_totals["total_time_loss_veh_h"]
                ),
                "recovered": terminated,
                "operation": update_result.operation,
                "management_changed": management_changed,
                "active_measure_count": len(
                    management_state
                ),
                "active_measures_after_json": (
                    json.dumps(
                        management_state,
                        sort_keys=True,
                    )
                ),
            })

        next_state = self._get_state(
            metrics,
            pr_t,
        )

        if self.episode_logger is not None:
            self.episode_logger.log_state(
                time_s=float(
                    metrics["time_s"]
                ),
                decision_epoch=(
                    self.decision_epoch
                ),
                feature_names=(
                    self.state_builder.feature_names
                ),
                state_vector=next_state,
            )

        info = {
            "scenario_id": self.scenario.scenario_id,
            "time_s": float(metrics["time_s"]),
            "decision_epoch": self.decision_epoch,
            "flood_stage": metrics["flood_stage"],
            "pr_t": pr_t,
            "delta_pr": delta_pr,
            "reward_pr_component": (
                pr_progress_reward
            ),

            "action_switch_cost": (
                action_switch_cost
            ),

            "queue_growth_veh": (
                queue_growth
            ),

            "queue_growth_penalty": (
                queue_growth_penalty
            ),

            "terminal_reward": (
                terminal_reward
            ),
            "reward_total_step": reward,
            "action": action_result_dict,
            "recovered": terminated,
            "recovery_time_s": recovery_status.recovery_time_s,
            **impact_totals,
            "management_changed": management_changed,
            "active_measures": management_state,
        }

        if terminated or truncated:
            if self.episode_logger is not None:
                paths = self.episode_logger.finalize(
                    recovered=terminated,
                    truncated=truncated,
                    recovery_status=recovery_status.as_dict(),
                    final_time_s=float(metrics["time_s"]),
                    impact_totals=impact_totals,
                )
                info.update({
                    key: str(value)
                    for key, value in paths.items()
                })
            self.close()

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    def _get_segment_queue_lengths(
            self,
    ) -> dict[str, float]:

        if self.segment_monitor is None:
            raise RuntimeError(
                "SegmentStateMonitor is not initialized."
            )

        snapshot = self.segment_monitor.snapshot()

        return {
            segment_id: float(
                metrics["queue_length_veh"]
            )
            for segment_id, metrics
            in snapshot.items()
        }

    def _compute_queue_growth(
            self,
            current_queues: dict[str, float],
    ) -> float:

        if not self.last_segment_queue_lengths:
            return 0.0

        growth = 0.0

        for segment_id, current_queue in (
                current_queues.items()
        ):
            previous_queue = (
                self.last_segment_queue_lengths.get(
                    segment_id,
                    current_queue,
                )
            )

            growth += max(
                0.0,
                current_queue - previous_queue,
            )

        return float(growth)

    def _performance_ratio(self, metrics: dict) -> float:
        return compute_performance_ratio(
            current_performance=float(
                metrics["mean_speed_mps"]
            ),
            baseline_performance=(
                self.baseline.get_mean_speed(
                    float(metrics["time_s"])
                )
            ),
        )

    def _get_state(
            self,
            metrics: dict,
            pr_t: float,
    ) -> np.ndarray:

        if self.segment_monitor is None:
            raise RuntimeError(
                "SegmentStateMonitor is not initialized."
            )

        if self.tactic_manager is None:
            raise RuntimeError(
                "TacticManager is not initialized."
            )

        if self.flood_engine is None:
            raise RuntimeError(
                "FloodEngine is not initialized."
            )

        segment_metrics = (
            self.segment_monitor.snapshot()
        )

        management_vector = (
            self.state_builder
            .management_slot_vector(
                active_measures=(
                    self.tactic_manager
                    .active_measures
                )
            )
        )

        return self.state_builder.build(
            current_time_s=float(
                metrics["time_s"]
            ),
            global_metrics=metrics,
            segment_metrics=segment_metrics,
            scenario=self.scenario,
            flood_stage=str(
                metrics["flood_stage"]
            ),
            flood_edge_ids=(
                self.flood_engine.edge_ids
            ),
            management_vector=(
                management_vector
            ),
            pr_t=pr_t,
        )

    def close(self) -> None:
        if self._sumo_running:
            try:
                if self.tactic_manager is not None:
                    self.tactic_manager.clear()
                if self.flood_engine is not None:
                    self.flood_engine.restore_base_state()
            finally:
                self.traci.close()
                self._sumo_running = False
