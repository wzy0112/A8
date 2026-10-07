# Step 8 — curriculum learning and flood domain randomization

## Objective

Move from a policy trained on one fixed Moderate flood to a policy that sees:

1. fixed Moderate;
2. a mixture of the three validated fixed scenarios;
3. continuously perturbed Light, Moderate and Severe scenarios.

Traffic demand remains fixed in Step 8. Demand phase and demand scaling are
reserved for the next extension.

## Randomized variables

For every episode in the randomized stage:

- severity template: Light, Moderate or Severe
- SUMO seed
- flood onset time
- onset-stage duration
- peak-stage duration
- recovery-stage duration
- peak speed parameter

The flood mechanism and validated flood site remain tied to their severity
template:

- Light: B471 speed reduction
- Moderate: A8 lane closure
- Severe: A8/B471 full closure

## Curriculum stages

Default:

| Stage | Sampling | PPO timesteps |
|---:|---|---:|
| 1 | fixed Moderate | 20,000 |
| 2 | mixed fixed Light/Moderate/Severe | 40,000 |
| 3 | continuously randomized scenarios | 100,000 |

The same PPO model is continued across stages.

## Install

    pip install -r requirements.txt

## 1. Test the sampler

    python smoke_test_randomization.py

Expected:

    Scenario randomization smoke test passed.

## 2. Small curriculum test

Run this before a long experiment:

    python train_curriculum.py ^
      --n-envs 1 ^
      --stage1-steps 2000 ^
      --stage2-steps 3000 ^
      --stage3-steps 5000 ^
      --seed 42

## 3. Full curriculum

    python train_curriculum.py ^
      --n-envs 4 ^
      --stage1-steps 20000 ^
      --stage2-steps 40000 ^
      --stage3-steps 100000 ^
      --seed 42

Do not use SUMO-GUI during parallel training.

## Output

    D:\SUMO_A9_Project\validation_results\
      step8_curriculum\seed_42\

Key files:

- `ppo_curriculum_final.zip`
- `curriculum_manifest.json`
- `stage_1_fixed_moderate\ppo_stage_model.zip`
- `stage_2_mixed_fixed\ppo_stage_model.zip`
- `stage_3_randomized\ppo_stage_model.zip`
- one `vec_normalize.pkl` per stage
- monitor, TensorBoard and callback metrics per stage

For final evaluation, use:

    stage_3_randomized\vec_normalize.pkl

because its observation statistics reflect the randomized training
distribution.

## Generalization evaluation

Example:

    python evaluate_generalization.py ^
      --run-dir ^
      D:\SUMO_A9_Project\validation_results\step8_curriculum\seed_42 ^
      --normalization ^
      D:\SUMO_A9_Project\validation_results\step8_curriculum\seed_42\stage_3_randomized\vec_normalize.pkl ^
      --episodes 30 ^
      --seed 10000

This runs PPO and No-op on exactly matched randomized episodes.

Output:

    D:\SUMO_A9_Project\validation_results\
      step8_generalization\generalization_episodes.csv

## Acceptance criteria

Technical success:

1. the sampler creates valid, varied timelines;
2. Light, Moderate and Severe all occur;
3. every reset creates a new scenario in randomized mode;
4. PPO continues from one curriculum stage to the next;
5. each stage saves a model and VecNormalize state;
6. the final model loads with the stage-3 normalization;
7. matched PPO/No-op randomized evaluation completes.

Performance success:

- recovery rate should not collapse on any severity;
- median recovery time should improve relative to No-op;
- gains should not come solely from unsafe lane opening;
- CO2, hard braking, queue exposure and time loss should be reviewed
  separately;
- results should be summarized by severity, not only as one global mean.

## Important limitation

Step 8 randomizes flood conditions but not traffic demand. The `X` field still
describes a fixed demand context. Demand phase, scale, truck percentage and
weather context should be introduced only after this curriculum is stable.
