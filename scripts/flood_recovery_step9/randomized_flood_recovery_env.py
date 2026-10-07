from __future__ import annotations

import random
from pathlib import Path

from flood_recovery_env import FloodRecoveryEnv
from recovery_metrics import RecoveryConfig
from scenario_sampler import FloodScenarioSampler

# evaluation能够看到non_peak peak

class RandomizedFloodRecoveryEnv(FloodRecoveryEnv):
    """
    Step 8 environment.

    One new flood scenario is sampled at every reset. All Step 7 observation,
    tactic, reward, recovery and logging behavior is retained.

    Scenario randomness and SUMO stochasticity use two independent RNG streams
    so each episode can change both while remaining reproducible.
    """

    def __init__(
        self,
        *,
        sampling_mode: str = "randomized",
        scenario_seed: int = 10_000,
        severity_weights: tuple[float, float, float] = (
            1.0,
            1.0,
            1.0,
        ),
        output_dir: Path | None = None,
        **kwargs,
    ):
        # Save the initial SUMO seed before forwarding kwargs to the Step 7 env.
        initial_sumo_seed = int(kwargs.get("sumo_seed", 42))

        # Base initialization requires one fixed template. It is replaced
        # before every SUMO reset.
        super().__init__(
            scenario_name="moderate",
            output_dir=output_dir,
            **kwargs,
        )

        self.sampling_mode = sampling_mode
        self.scenario_seed = int(scenario_seed)

        # Independent random streams:
        # - scenario_rng controls flood scenario sampling.
        # - sumo_rng controls the SUMO seed used by each episode.
        self.scenario_rng = random.Random(
            self.scenario_seed
        )
        self.sumo_rng = random.Random(
            initial_sumo_seed + 200_000
        )

        self.scenario_sampler = FloodScenarioSampler(
            mode=sampling_mode,
            severity_weights=severity_weights,
        )
        self.episode_index = 0

    def _scenario_info(self) -> dict:
        return {
            "sampling_mode": self.sampling_mode,
            "sampled_severity": self.scenario.severity,
            "sampled_site_id": self.scenario.site_id,
            "sampled_mechanism": self.scenario.mechanism.value,
            "sampled_demand_phase": self.scenario.demand_phase,
            "sampled_onset_t": self.scenario.onset_t,
            "sampled_peak_start_t": self.scenario.peak_start_t,
            "sampled_peak_end_t": self.scenario.peak_end_t,
            "sampled_recovery_end_t": self.scenario.recovery_end_t,
            "sampled_peak_speed_mps": self.scenario.peak_speed_mps,
            "sampled_closed_lane_indices": list(
                self.scenario.closed_lane_indices
            ),
            "sampled_max_sim_time_t": self.scenario.max_sim_time_t,
            "sampled_sumo_seed": self.sumo_seed,
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Re-seed both streams independently.
            base_seed = int(seed)
            self.scenario_seed = base_seed + 100_000
            self.scenario_rng.seed(
                self.scenario_seed
            )
            self.sumo_rng.seed(
                base_seed + 200_000
            )

        # IMPORTANT: update the SUMO seed on every episode, not only when
        # Gym explicitly passes reset(seed=...).
        self.sumo_seed = self.sumo_rng.randint(
            0,
            2_147_483_647,
        )

        self.episode_index += 1
        self.scenario = self.scenario_sampler.sample(
            self.scenario_rng,
            episode_index=self.episode_index,
        )

        # Recovery search must follow the sampled peak_end_t.
        self.recovery_config = RecoveryConfig(
            threshold=self.recovery_config.threshold,
            stability_window_s=(
                self.recovery_config.stability_window_s
            ),
            search_start_t=float(
                self.scenario.peak_end_t
            ),
        )

        observation, info = super().reset(
            seed=None,
            options=options,
        )
        info.update(self._scenario_info())
        return observation, info

    def step(self, action):
        # The Step 7 base env builds the normal terminal/info dictionary.
        # Add the sampled scenario fields to EVERY returned info so that
        # VecEnv evaluation can recover them at episode end.
        observation, reward, terminated, truncated, info = (
            super().step(action)
        )
        info.update(self._scenario_info())

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
