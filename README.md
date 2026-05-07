# HMRD-PPO: Heterogeneous Multi-Robot Deployment via Proximal Policy Optimization

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

## 📖 Introduction

**HMRD-PPO-Robotic-Security** is a Multi-Agent Reinforcement Learning (MARL) framework designed for the co-optimization of heterogeneous robotic security assets. Specifically, it focuses on the optimal deployment of surveillance cameras and capture drone launch stations against adaptive adversarial drones.



https://github.com/user-attachments/assets/e8cab1a2-529b-44ea-b06c-62a7ae2f44a9


## 🚀 Key Features

- **Cooperative MARL Architecture**: Centralized fleet controllers (PPO) managing continuous spatial actions.
- **Curriculum Learning**: Six escalating difficulty levels of adversarial behaviors (from static paths to fully adaptive, memory-based routing).
- **Automated Red Teaming**: AAA* search with dynamic penalty zones serves as a deterministic, reproducible falsification tool.
- **MCTS Baseline**: Includes an Iterative Local-Search Monte Carlo Tree Search algorithm for spatial optimization benchmarking.
- **Continuous State & Action Spaces**: Parameterized Gaussian policies handling complex, real-valued coordinate maps.

## 📁 Project Structure

```text
@HMRD-PPO-Robotic-Security/
├── config.yaml                     # Master configuration for simulation, evaluation, and training
├── data/                           # Initialization data
│   ├── roi.xlsx                    # Regions of Interest (RoI) definitions
│   └── placement.xlsx              # Initial placements for Launch Pads and Cameras
├── results/                        # Output artifacts
│   └── saved_models/               # Trained PyTorch model weights (.pth)
├── scripts/                        # Executable top-level scripts
├── src/                            # Core source code
│   ├── algorithms/                 # Learning and decision-making logic (PPO, MCTS, AAA*, rewards)
│   ├── simulation/                 # Physics, interception mechanics, and parallel execution engine
│   └── utils/                      # Geometry calculations and file I/O
└── tests/                          # Unit and integration tests
```

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Itamar-Gold/HMRD-PPO-Robotic-Security.git
   cd HMRD-PPO-Robotic-Security
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

All configuration parameters (learning rates, simulation constraints, curriculum levels) are managed centrally in `config.yaml`.

### 1. Train HMRD-PPO (Simultaneous)
To train both the CDFC and CFC concurrently across the curriculum levels (This is the best-performing variant):
```bash
python scripts/train_hmrd_ppo_simultaneous.py
```

### 2. Train HMRD-PPO (Alternating)
To train the agents using the alternating freezing strategy:
```bash
python scripts/train_hmrd_ppo_alternating.py
```

### 3. Run MCTS Baseline
To run the Monte Carlo Tree Search spatial optimization algorithm:
```bash
python scripts/run_mcts.py
```

### 📊 Results & Artifacts
Outputs including JSON logs, optimal deployment configurations (`best_configs{level}.json`), performance plots, and trained `.pth` models will be saved automatically to the `results/` directory.

## 📝 Terminology Guide

- **Red Team / Adversary**: Intruder drones attempting to reach targets around RoI.
- **Blue Team**: The defending robotic security system.
- **Launch Pads**: Active stations that launch capture drones to intercept the Red Team (Controlled by CDFC).
- **Cameras**: Static surveillance sensors providing early warning detection (Controlled by CFC).
- **CDFC**: Capture Drone Launch Stations Fleet Controller.
- **CFC**: Cameras Fleet Controller.
