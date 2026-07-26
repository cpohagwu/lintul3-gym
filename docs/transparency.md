# Transparency

Every `step()` returns a `gym.spaces.Dict` observation with `crop`, `weather`, and `management`
kept as separately-labeled groups -- not one opaque flattened vector -- plus an `info` dict
carrying the weather source/location/year, growth, nitrogen cost, reward, and a serializable
`EpisodeRecord`. `env.unwrapped.history` preserves the full episode sequence: nothing about a
transition is hidden from the caller.

## Reference policies (`lintul3_gym.policies`)

- **`ZeroNitrogenPolicy`** -- never applies nitrogen; a no-input lower-bound baseline.
- **`ExpertPolicy`** -- reproduces the 10 and 5 g N/m² applications from the bundled spring-wheat
  reference `.agro` (Apr 10, May 5).
- **`StandardPracticePolicy`** -- reproduces the Kallenberg et al. (2023) "Standard Practice"
  baseline for winter wheat: a configurable total nitrogen dose split across three real
  fertilization dates (Feb 24, Mar 26, Apr 29).

`ExpertPolicy` and `StandardPracticePolicy` are both *fixed calendar-dose* schedules -- apply a
predetermined amount on predetermined dates, regardless of crop state -- sharing one internal
date-window helper rather than duplicating that logic.

## Evaluation helpers

- **`evaluate_policy(environment, policy, seed=...)`** -- runs one episode (a `Policy`, or a
  Stable-Baselines3 model) and returns a complete `EvaluationResult` (`total_reward`, `final_wso`,
  `total_nitrogen`, and the full `history`).
- **`evaluate_policy_over_weather(environment, policy)`** -- runs `evaluate_policy` once per
  `(location, year)` combination in a `WeatherConfig`, round-robin, so results from different
  policies line up one-to-one by combination.
- **`evaluate_sb3_policy(vec_env, model, n_episodes=...)`** -- the `VecNormalize`-aware
  counterpart, reading each step's `info["record"]` directly (a `VecEnv` auto-resets before
  `.history` can otherwise be read back).

Use these together with the plotting helpers in `lintul3_gym.viz` (see [Usage](usage.md)) to
compare a trained agent's behavior against the reference policies above -- every figure is built
from the same transparent `EpisodeRecord` data the agent itself received, not a separate summary
computed after the fact.
