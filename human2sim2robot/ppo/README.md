# PPO

The purpose of this repository is to provide a (relatively) simple codebase for reinforcement learning with PPO for continuous control tasks. It is heavily inspired by [rl_games](https://github.com/Denys88/rl_games/) and [minimal-stable-PPO](https://github.com/ToruOwO/minimal-stable-PPO).

## Key Changes

`rl_games` is very hard to understand, modify, and debug due to its generality and heavy use of builders, factories, and subclasses. `minimal-stable-PPO` is much simpler, but it is missing some essential features like asymmetric actor critic.

Additional changes:

- Config validation to ensure that all necessary parameters are provided and extra/incorrectly spelled parameters are not provided (done using dataclasses with defaults + a dict_to_dataclass function)

- Renamed so that `PpoAgent` is the actor + critic trained with PPO and `PpoPlayer` is the actor that is used for inference (not training)

- Renamed `CentralValue` to `AsymmetricCritic` for clarity

- Removed unused features like discrete algorithms, self-play, image-based RL, etc.