from __future__ import annotations

import argparse
import sys
import traceback
import numpy as np

from flood_recovery_env import FloodRecoveryEnv


def run_backend(backend: str, scenario: str, epochs: int, gui: bool) -> bool:
    print("=" * 72)
    print(f"PART 6 SMOKE TEST | backend={backend!r} | gui={gui}")
    print("=" * 72)

    env = None
    try:
        try:
            env = FloodRecoveryEnv(
                scenario_name=scenario,
                decision_period_s=300,
                gui=gui,
                sumo_backend=backend,
                enable_episode_logging=False,
                policy_version="part6_backend_smoke_test",
            )
        except TypeError as exc:
            if "sumo_backend" in str(exc):
                raise RuntimeError(
                    "FloodRecoveryEnv does not yet accept `sumo_backend`. "
                    "Apply the Part 6 backend refactor first."
                ) from exc
            raise

        print("[1/5] Environment created.")

        obs, info = env.reset()
        print("[2/5] reset() completed.")
        print("      obs.shape:", obs.shape)
        print("      state_dim:", info.get("state_dim"))
        print("      time_s:", info.get("time_s"))
        print("      PR(t):", info.get("pr_t"))

        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))
        print("[3/5] Initial observation checks passed.")

        hold_action = np.array([0, 0, 0], dtype=np.int64)

        for epoch in range(1, epochs + 1):
            obs, reward, terminated, truncated, info = env.step(hold_action)

            assert obs.shape == env.observation_space.shape
            assert np.all(np.isfinite(obs))
            assert np.isfinite(float(reward))

            print(
                f"[4/5] epoch={epoch} "
                f"time_s={info.get('time_s')} "
                f"reward={float(reward):.6f} "
                f"PR={info.get('pr_t')} "
                f"recovered={terminated} "
                f"truncated={truncated}"
            )

            if terminated or truncated:
                break

        print("[5/5] Backend smoke test PASSED.")
        return True

    except Exception:
        print("\nSMOKE TEST FAILED")
        traceback.print_exc()
        return False

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Part 6 backend smoke test for FloodRecoveryEnv."
    )
    parser.add_argument(
        "--backend",
        choices=("traci", "libsumo", "both"),
        default="both",
    )
    parser.add_argument(
        "--scenario",
        choices=("light", "moderate", "severe"),
        default="moderate",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Only valid with --backend traci.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    if args.gui and args.backend in {"libsumo", "both"}:
        parser.error("--gui can only be used with --backend traci.")

    backends = (
        ["traci", "libsumo"]
        if args.backend == "both"
        else [args.backend]
    )

    all_ok = True
    for backend in backends:
        all_ok = run_backend(
            backend=backend,
            scenario=args.scenario,
            epochs=args.epochs,
            gui=args.gui,
        ) and all_ok

    if all_ok:
        print("\nPART 6 BACKEND SMOKE TEST PASSED")
        return 0

    print("\nPART 6 BACKEND SMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
