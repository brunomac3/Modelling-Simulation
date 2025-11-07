## 1. Topic

This project focuses on the simulation of highway driving scenarios to train and evaluate an autonomous vehicle (ego car) capable of performing maneuvers such as lane changing, merging, and overtaking in dynamic traffic environments. The study leverages reinforcement learning to enable the ego vehicle to make sequential decisions that balance safety, efficiency, and compliance with traffic regulations. Through interaction with a simulated environment, the RL agent learns optimal driving behaviors that adapt to varying traffic densities and vehicle behaviors.


## 2. Problem Formulation

The problem addressed in this study is the design and implementation of a simulation framework that enables an autonomous vehicle to learn safe and efficient driving strategies in realistic, dynamic traffic environments. The system must account for diverse traffic conditions, including varying vehicle densities, aggressiveness levels, and speed distributions, to ensure robustness and adaptability of the learned policy.

The main challenge lies in enabling the reinforcement learning agent (ego vehicle) to make effective driving decisions - such as when to change lanes, accelerate, or brake - while minimizing the risk of collisions and maintaining compliance with road rules. The simulation environment must therefore balance realism, controllability, and computational efficiency to allow for effective model training and evaluation.

The ultimate goal is to optimize the vehicle’s driving policy for both safety (avoiding collisions and unsafe maneuvers) and efficiency (maintaining smooth traffic flow and timely goal completion) under a range of traffic scenarios.

## 3. Objectives

The primary objectives of this project are:

- To train a reinforcement learning-based driving policy capable of performing safe, efficient, and compliant maneuvers in multi-vehicle highway scenarios.

- To evaluate the performance of the trained policy using key performance indicators (KPIs) such as safety metrics, efficiency indices, and comfort measures.

- To compare and analyze different reinforcement learning algorithms - such as Proximal Policy Optimization (PPO), Deep Q-Network (DQN), and Soft Actor-Critic (SAC) - to determine their relative effectiveness in autonomous driving tasks.

These objectives align with the broader goal of developing decision-support models that enhance the reliability and performance of autonomous vehicles in realistic, dynamic environments.


## 4. Main Entities  

- **Autonomous Vehicle (Ego Car)**  
  - The reinforcement learning agent we are training.  
  - Makes driving decisions (lane changing, merging, overtaking).  
  - Learns through interaction with the environment.  

- **Other Vehicles (Traffic Environment)**  
  - Surrounding cars driven by predefined behavior models.  
  - Provide dynamic obstacles and interactions.  
  - Influence task difficulty (traffic density, aggressiveness, speed distribution).  

- **Road/Highway Environment**  
  - Multi-lane highway with traffic flow.  
  - Defines constraints: lanes, speed limits, road length, entry/exit ramps.  
  - Provides sensory input to the ego vehicle (positions, velocities).  

(**Note**: while the RL algorithm, reward function, and evaluation metrics play a central role in how the simulation operates, they are considered modeling components rather than entities of the traffic system itself.)

---

## 5. Properties of Entities  

| Entity         | Properties                                                                                  | Notes                           |  
| -------------- | ------------------------------------------------------------------------------------------- | ------------------------------- | 
| Ego Vehicle    | Position, velocity, acceleration, lane index, action space (lane-change, accelerate, brake) | Controlled by RL agent          |    
| Other Vehicles | Position, velocity, lane, driving policy                                                    | Define traffic dynamics         |     
| Road/Highway   | Number of lanes, length, speed limits, entry/exit ramps                                     | Defines environment constraints |     

---

## 6. Model Type  

This simulation is primarily **Prescriptive**, since the goal is to design and evaluate strategies (via reinforcement learning policies) that autonomous vehicles can use to act safely and efficiently in traffic scenarios such as lane changing, merging, and overtaking.  

It also has a **Predictive aspect**, as the trained agent implicitly forecasts the outcomes of its possible actions (whether a lane change will lead to a safe maneuver or a collision) in order to maximize long-term rewards.  

It is not **Descriptive**, because the model is not focused on replicating current human driver behavior. Instead, it seeks to propose better strategies. Nor is it purely **Speculative**, since the experiments are grounded in realistic traffic environments and established simulation frameworks.  


## 7. Metrics


| **Entity** | **Metrics** | **Description** | **Notes** |
|-------------|--------------|------------------|------------|
| **Ego Vehicle** | **Collision Rate** | Fraction of episodes ending in collision. Measures driving safety. | Lower is better; key for safety evaluation. |
|  | **Average Speed** | Mean longitudinal velocity over an episode. | Indicates efficiency; should respect speed limits. |
|  | **Lane Change Frequency** | Number of lane changes per episode. | Too high → erratic driving; too low → passive driving. |
|  | **Reward per Episode** | Total accumulated reward per episode. | Global performance indicator for RL agent. |
|  | **Speed Limit Compliance** | Fraction of time speed ≤ limit. | Reflects adherence to road rules. |
|  | **Time-to-Collision (TTC)** | Minimum time before potential collision. | Safety measure; can be averaged per episode. |
|  | **Episode Duration** | Time until success or termination. | Lower means efficient, but not at cost of safety. |
| **Other Vehicles** | **Traffic Density** | Number of vehicles per kilometer or per lane. | Affects task difficulty; can be parameterized. |
|  | **Average Relative Speed** | Mean speed difference between ego and surrounding vehicles. | Influences overtake/merge difficulty. |
|  | **Aggressiveness Index** | Tendency to accelerate/brake abruptly or follow too closely. | Defines behavior model complexity. |
|  | **Traffic Flow Stability** | Standard deviation of vehicle speeds in the scene. | Lower = smoother traffic; high = more chaotic. |
| **Road/Highway** | **Lane Utilization Ratio** | Percentage of time ego uses each lane. | Shows strategic lane use and distribution. |
|  | **Speed Limit Compliance (Global)** | % of vehicles within legal speed range. | Indicates realism and rule adherence of simulation. |
|  | **Average Throughput** | Number of vehicles passing a given point per unit time. | Reflects efficiency of entire environment. |
|  | **Road Occupancy Rate** | Portion of road length occupied by vehicles. | Used to measure congestion level. |
|  | **Scenario Completion Rate** | Fraction of successful runs (goal reached, no collision). | High completion = stable, effective simulation setup. |


##  8. Indicators

| **Indicator** | **Definition / Formula** | **What It Measures** | **Notes** |
|----------------|---------------------------|----------------------|------------|
| **Safety Index (SI)** | Combines collision rate, TTC, and safety violations:<br>`SI = w1*(1 - CollisionRate) + w2*(avg(TTC_norm)) + w3*(1 - ViolationRate)` | Overall driving safety and risk avoidance. | High SI → safer driving policy; weights tuned per experiment. |
| **Efficiency Index (EI)** | Weighted combination of average speed and completion rate:<br>`EI = w1*(avg_speed / speed_limit) + w2*(CompletionRate)` | Measures traffic efficiency and goal achievement. | Balance between fast progress and task success. |
| **Comfort Index (CI)** | Inverse of acceleration jerk and lane-change frequency:<br>`CI = 1 - norm(Jerk + α * LaneChangeFreq)` | Reflects smoothness and passenger comfort. | Higher = smoother, more human-like driving. |
| **Rule Compliance Index (RCI)** | Based on adherence to traffic rules:<br>`RCI = 1 - (SpeedViolations + DistanceViolations) / TotalTime` | How well the agent respects traffic regulations. | Penalizes overspeeding and unsafe following. |
| **Learning Efficiency (LE)** | `LE = PerformanceScore / TrainingSteps` | How quickly the RL model learns a stable, high-performing policy. | Allows comparison of RL algorithms (e.g., PPO vs SAC). |
| **Traffic Flow Index (TFI)** | `TFI = Throughput / Density` | Reflects how well the overall traffic moves given congestion. | High TFI → good coordination and lane distribution. |
| **Environment Stability Index (ESI)** | `ESI = 1 - std(vehicle_speeds) / mean(vehicle_speeds)` | Degree of stability and consistency in traffic flow. | Low variability indicates stable environment. |
| **Safety–Efficiency Trade-off (SET)** | `SET = β1*(SafetyIndex) + β2*(EfficiencyIndex)` | Evaluates balance between safe and efficient driving. | Useful for comparing different policies’ trade-offs. |
| **Global Performance Score (GPS)** | Aggregate score combining all KPIs:<br>`GPS = a*SI + b*EI + c*CI + d*RCI` | Single number summarizing overall system performance. | Weight coefficients can be tuned experimentally. |


## 9. Data Requirements

The simulation relies on both synthetic and benchmark traffic data to represent realistic driving conditions. The data required primarily concerns vehicle kinematics and traffic flow dynamics, which define how vehicles move and interact on a multi-lane highway.

**Type of Data:**
The simulation collects and processes information on vehicle position, velocity, acceleration, lane index, and inter-vehicle distances. Traffic flow variables such as vehicle density, average speed, and lane occupancy are also used to characterize the environment and evaluate system performance.

**Sources:**
Data is obtained from synthetic simulations using platforms such as HighwayEnv (Python).

**Assumptions:**
All vehicles comply with physical kinematic constraints (e.g., maximum acceleration and braking limits). Weather and road conditions remain constant within each simulation episode to isolate learning effects. Sensor perception is assumed to be ideal, meaning no measurement noise or occlusion is introduced at this stage of development.

## 10. Methods & Tools

To implement and evaluate the autonomous driving simulation, a combination of open-source simulation environments, reinforcement learning libraries, and analysis tools are employed.

**Simulation Tools:**
The project utilizes HighwayEnv for generating realistic highway traffic scenarios. These platforms provide controllable conditions for defining lane configurations, traffic densities, and vehicle behaviors, ensuring reproducible experiments.

**Programming Environment:**
Development and experimentation are conducted in Python 3.10, leveraging the Gymnasium interface for consistent agent–environment interactions. This setup allows modular testing of different control policies and reward functions.

**Evaluation and Visualization:**
Performance metrics and training progress are monitored through TensorBoard and custom data analysis scripts.