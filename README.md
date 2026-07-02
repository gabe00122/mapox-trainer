# Partially Observable Multi-Agent RL with Transformers

This project provides a JAX-based framework for training Transformer-based agents in custom, partially observable multi-agent reinforcement learning (MARL) environments. The custom grid environments are designed to test memory, planning, multi-task learning and cooperation.

https://github.com/user-attachments/assets/cf2a4c5b-37ce-4cc5-a63f-404277562152

Blog Posts:
[Online Transformer RL](https://valm.gabrielkeith.pages.dev/posts/transformer_rl)
[Memory and Implicit Communication](https://gabrielkeith.dev/posts/memory)


### ✨ Key features

* **Transformer-based Agents**: Uses a Transformer-over-time architecture to handle partial observability. Uses updated transformer architecture such as RoPE and pre-layer norm.
* **High Performance**: Two-layer transformers train at **3.8 million+** steps per second and an 8-layer transformer trains at **1.36 million** steps per second on a single 5090. cuDNN, grouped query attention, and bfloat16 training significantly boost performance.
* **Custom Environments**: Includes several challenging MARL environments designed to test memory and coordination.
* **PPO Implementation**: A clean and efficient Proximal Policy Optimization (PPO) implementation.
* **Multi-task training**: One model can be trained on multiple environments simultaneously with unique task ID embeddings.
* **Snapshot League self play**: Policy snapshots can be rotated into the opponent pool during training.
* **TrueSkill Evaluations**: Policies for zero-sum games can be compared using a TrueSkill-based evaluation system and a model checkpoint pool.
* **Discretized Value Functions**: Option to either minimize MSE or HL-Gauss discretized values.

---

## 🚀 Getting started

### Prerequisites

* Python 3.13+
* [uv](https://github.com/astral-sh/uv) (a fast Python package installer and resolver)
* [FFmpeg](https://ffmpeg.org/) (required for recording videos)

### Installation

Clone the repository and install the dependencies:

```bash
git clone git@github.com:gabe00122/mapox-trainer.git
cd mapox-trainer
uv sync --extra cuda
```


## 💻 Usage

### Download the pretrained model

Install hf cli: https://huggingface.co/docs/huggingface_hub/guides/cli
```bash
hf download gabe00122/mapox-checkpoint --local-dir ./results
```

This downloads `multitask`, a pretrained model that supports the following environments: `return`, `koth`, `prey`, and `scouts`.

### Watch a trained agent
To render an environment with a trained agent, use the enjoy command with the run name.

```bash
uv run pmarl enjoy multitask --seed 5 --env return --video-path out.mp4
```

### Play an environment yourself
You can control an agent in the environment yourself using the `pov` and `human` options for enjoy.

```bash
uv run pmarl enjoy multitask --env return --pov --human
```

Typical control scheme
Keyboard:
w: up
d: right
s: down
a: left
e: dig a wall
space: attack
n: cycle to the next agent

### Train an agent
To start a training run, use the train command and provide a configuration file.

```bash
uv run pmarl train --config ./config/multitask.json
```
A unique run name will be generated for you.
To view recently trained models you can use:

```bash
uv run pmarl runs
```

### Configuration

Training is configured via JSON files. See `config/` for examples. The key sections are:

**Top-level**
| Field | Description |
|---|---|
| `seed` | Random seed, or `"random"` to generate one |
| `num_envs` | Number of parallel environments (for single-environment configs) |
| `max_env_steps` | Maximum steps per episode |
| `update_steps` | Total number of PPO updates to run |
| `num_checkpoints` | Number of model checkpoints to save during training (default: 50) |
| `snapshot_league` | Enable snapshot league self-play (default: false) |

**Model** (`learner.model`)
| Field | Description |
|---|---|
| `num_layers` | Number of transformer layers |
| `hidden_features` | Embedding dimension |
| `layer.history.type` | Sequence model: `"attention"` or `"rnn"` |
| `layer.history.num_heads` / `num_kv_heads` | Query heads and key-value heads (for grouped query attention) |
| `layer.feed_forward.size` | Feed-forward hidden size |
| `layer.feed_forward.glu` | Use gated linear units (default: true) |
| `value.type` | Value function head: `"mse"` or `"hl_gauss"` |
| `dtype` | Compute dtype: `"float32"`, `"bfloat16"`, or `"float16"` |

**Optimizer** (`learner.optimizer`)
| Field | Description |
|---|---|
| `type` | `"adamw"` or `"muon"` |
| `learning_rate` | Learning rate |
| `max_norm` | Gradient clipping max norm |

**PPO** (`learner.trainer`)
| Field | Description |
|---|---|
| `epoch_count` | PPO epochs per update |
| `minibatch_count` | Number of minibatches per epoch |
| `discount` | Discount factor (gamma) |
| `gae_lambda` | GAE lambda |
| `entropy_coef` | Entropy bonus coefficient |

**Environment** (`environment`)

For a single environment, set `env_type` directly (e.g. `"find_return"`, `"king_hill"`, `"scouts"`). For multi-task training, set `env_type` to `"multi"` with an `envs` array — see `config/multitask.json` for an example.

---

### Evaluating Zero-Sum Games

To track progress and compare different training runs for zero-sum games, we can use TrueSkill to create an Elo-like score using an opponent pool of model checkpoints.

To build a TrueSkill graph from a series of runs, use the following command.

```bash
uv run pmarl eval --run blue-whale --run red-fish --rounds 1000 --out ./analysis/graph.png
```

---

### Multi-task Training

If the observation and action spaces are shared across environments, then one policy can be trained on multiple environments simultaneously.
See the `multitask.json` config for an example.

If multitask training was used, the `--env name` argument is required for evaluation and playback to specify which task you are viewing or evaluating. For non-multitask environments this option is unnecessary.

## 🏋️‍♂️ Environments

The environments were originally developed as part of this project and have been extracted into a standalone package so they can be used independently: https://github.com/gabe00122/mapox

### Grid Return

A multi-agent 2D grid world task requiring spatial memory.

* Description: One or more goal tiles are placed at random locations. When an agent finds a goal, it receives a `+1` reward, and the agent is moved to a new random location. Agents must remember goal locations and navigate around obstacles to return to them efficiently. The number of goals can be configured via `num_flags` and defaults to one.

Agents can "dig" through obstacle tiles. Moving into an obstacle removes the tile but adds a timeout before the agent can move again.

https://github.com/user-attachments/assets/257f65fe-7c54-4879-8ea0-2d744dcc65f2

### Scouts
A multi-agent coordination task with two specialized agent types. A Harvester must first unlock a resource tile, which a Scout can then gather.

Scout: Fast-moving agents that can gather resources only after they are "unlocked."
Harvester: Slow agents (can only move every 6th turn). When a Harvester reaches a resource tile, it gets a reward and unlocks the resource, allowing Scouts to gather it.

https://github.com/user-attachments/assets/3d0cfc40-f0fd-47a0-bd14-fc5e7b0af968

### Traveling Salesman

Several waypoints are randomly scattered across the map. Agents receive a reward the first time they reach each waypoint. When all waypoints have been collected, they reset.

https://github.com/user-attachments/assets/af009d24-c65e-4195-99af-0a4e703652cd

### King of the Hill

A multi-agent gridworld where two teams of Knights and Archers battle to capture and hold flags.

* Randomly generated maps with destructible walls and central control points
* Knights: melee fighters with higher HP
* Archers: ranged fighters with arrows and cooldowns
* Teams score points every turn for each flag they control
* Agents can move, attack, dig through walls, or fire arrows
* Rewards are fully team-shared, encouraging coordination

https://github.com/user-attachments/assets/b3a3810b-41a5-4d49-b851-00d305c200f6

## Scaling Results

Performance scales predictably with network depth.
<img width="956" height="400" alt="image" src="https://github.com/user-attachments/assets/698dc9d9-3a37-4959-aa2c-5e9f06e3ba59" />

## Craftax

The craftax environment is also supported: https://github.com/MichaelTMatthews/Craftax

20.8% reward with 1b samples.
Episodes need to be truncated to fit within the context window, I truncated to 1024 step episodes.

https://github.com/user-attachments/assets/b6c40151-6012-4930-89af-928bea54352b

---

## 🙏 Acknowledgements

This project was made possible thanks to:

* **Google Research Cloud TPU Program** — for providing access to TPUs that enabled training and experimentation.  
* **[Urizen Onebit Tileset](https://vurmux.itch.io/urizen-onebit-tileset)** by Vurmux — for the excellent pixel art tileset used in the gridworld environments.
