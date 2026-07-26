# LINTUL3-Gym documentation

LINTUL3-Gym is a transparent, reproducible [Gymnasium](https://gymnasium.farama.org/) environment
for nitrogen-management reinforcement learning, built on PCSE's LINTUL3 crop model. These pages
cover the package's configuration surface in more depth than the root README -- start there for
installation and a quick-start example.

- [Usage](usage.md) -- environment construction, weather sources, and the Stable-Baselines3
  adapters.
- [Custom environments](custom-environments.md) -- pointing `Lintul3Env` at a different crop.
- [Transparency](transparency.md) -- what every transition exposes, and the reference policies
  used to evaluate a trained agent against.

See the root [README](../README.md) for the full example list and how to cite this software.
