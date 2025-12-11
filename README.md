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

--- 
## Autonomous Balancing System: Perception–Action Loop (PPO-Based Control)

The autonomous bicycle operates using a high-frequency **Perception–Action loop**, where decisions are made continuously based on visual and telemetry data. Unlike PID controllers that react to error, the PPO Agent **predicts** optimal actions for stability, navigation, and forward motion.

---

## 📡 1. Observation

The robot collects information through two synchronized data streams to build a complete understanding of its environment:

### Visual Input
- A **240×240 RGB image** from the onboard camera.  
- Provides perception of the horizon, tilt, obstacles, and relative target position.

### Telemetry Data
A **16-dimensional sensor vector**, containing:
- Lidar distances for real-time obstacle detection  
- Orientation data (roll, pitch, yaw)  
- Linear and angular velocity  

These two inputs collectively form the full system state used for decision-making.

---

## 🧠 2. Decision (Actor Network Outputs)

The fused sensory data is passed into the **Actor network**, which generates continuous control signals for the actuator system. The network simultaneously regulates:

### Steering
- Adjusting steering **velocity** and **force** for directional control.

### Throttle
- Controlling **rear wheel velocity** to maintain forward momentum.

### Balancing
- Actively managing the **pendulum position (balancing mass)** to generate counter-torque, ensuring upright stability during motion.

These three control outputs work together to maintain balance and navigate toward the target.

---

## 📈 3. Learning (Critic + PPO Optimization)

The system improves continuously through Reinforcement Learning:

### Evaluation
- The **Critic network** evaluates the quality of each action by estimating the expected return.

### Optimization
- The **PPO (Proximal Policy Optimization)** algorithm updates the policy using clipped objective functions.
- The reward function prioritizes:
  - Reducing distance to the target  
  - Maximizing stability (minimizing tilt, wobble, and unwanted oscillation)  

This creates a stable, self-learning control strategy capable of autonomous balancing without manual tuning.

---
