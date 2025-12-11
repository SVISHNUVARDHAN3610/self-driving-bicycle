<div align="center">

# 🚲Balancing-and-Navigation-of-a-Bicycle-using-Policy-Gradient-Reinforcement-Learning
### Reinforcement Learning with Multi-Modal Perception

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Simulation](https://img.shields.io/badge/Sim-PyBullet-blue?style=for-the-badge&logo=bullet&logoColor=white)](https://pybullet.org/)
[![Algorithm](https://img.shields.io/badge/RL-PPO%20(Actor--Critic)-green?style=for-the-badge)]()

> **"Learning to Ride."**
> An autonomous bicycle agent trained via Proximal Policy Optimization (PPO) to balance and navigate using a reaction pendulum and visual perception.

[Project Overview](#overview) • [System Architecture](#architecture) • [Neural Network](#network) • [Installation](#installation)

</div>

---

## <a name="overview"></a>🧐 Project Overview

Balancing a bicycle at low speeds is a classic control theory problem due to its inherent instability. This project solves that challenge using **Deep Reinforcement Learning**. 

Instead of traditional PID controllers, we train an AI agent in a high-fidelity **PyBullet** physics environment. The agent controls the steering torque and a **Reaction Pendulum** to maintain stability while navigating toward a target. It utilizes a **Multi-Modal Neural Network** that fuses raw camera feeds with LIDAR sensor data to make decisions.

### 🌟 Key Features
* **⚖️ Active Balancing:** Uses a reaction mass (pendulum) and steering manipulation to stay upright.
* **🧠 Hybrid Perception:** Processes **Visual Data** (Camera Feed) and **Telemetry** (Lidar/Velocity) simultaneously.
* **🤖 PPO Algorithm:** Implements Proximal Policy Optimization with an Actor-Critic architecture for stable training.
* **physics Simulation:** Custom PyBullet environment with realistic gravity, collision, and joint dynamics.

---

## <a name="architecture"></a>🏗️ System Architecture

The system closes the loop between the Physics Environment and the Intelligent Agent.
```mermaid
graph TD
    subgraph PyBullet Environment
    A[Bicycle Robot] -->|Camera Feed (RGB)| B(Visual Preprocessing)
    A -->|LIDAR & Velocity| C(Sensor Data)
    D[Physics Engine] --> A
    end

    subgraph PPO Agent
    B --> E[CNN Layers]
    C --> F[MLP Layers]
    E & F --> G{Feature Fusion}
    G --> H[Actor Network]
    G --> I[Critic Network]
    end

    H -->|Action: Steer & Pendulum| D
    I -->|Value Estimate| J[Advantage Calculation]
    J --> H
```

## Autonomous Balancing System: Perception–Action Loop (PPO Based)

The autonomous balancing system operates on a high-frequency **Perception–Action loop**.  
Unlike traditional PID controllers that react to errors, the **PPO Agent** proactively predicts optimal torque and control signals to maintain stability while navigating dynamic environments.

### 1. Observation Phase
The agent receives two types of sensory input:

- **240×240 RGB Image** from the onboard front-facing camera  
- **16-Dimensional Sensor Vector**, containing:
  - Lidar distance readings  
  - Orientation (roll, pitch, yaw)  
  - Linear and angular velocity  

These inputs are fused to form the state representation.

### 2. Decision Phase (Actor Network Output)

The Actor network generates **continuous control signals** for:

1. **Steering Velocity & Force**  
2. **Rear Wheel Velocity (Throttle Control)**  
3. **Pendulum Position Adjustment**  
   - Controls the **balancing mass** to maintain equilibrium  

The output is optimized to ensure smooth navigation and dynamic stability.

### 3. Learning Phase (Critic + PPO)

- The **Critic network** evaluates the quality of each action by estimating the value function.  
- The **PPO (Proximal Policy Optimization)** algorithm updates the policy using clipped surrogate objectives.  
- The reward function is designed to maximize:
  - Forward progress toward the target  
  - Stability (reduced oscillation, minimal tilt)  
  - Smooth control effort  

This creates a robust, self-learning control architecture capable of real-time balancing without manual tuning or PID heuristics.
