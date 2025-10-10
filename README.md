
## 1. Main Entities  

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

## 2. Properties of Entities  

| Entity         | Properties                                                                                  | Notes                           |  
| -------------- | ------------------------------------------------------------------------------------------- | ------------------------------- | 
| Ego Vehicle    | Position, velocity, acceleration, lane index, action space (lane-change, accelerate, brake) | Controlled by RL agent          |    
| Other Vehicles | Position, velocity, lane, driving policy                                                    | Define traffic dynamics         |     
| Road/Highway   | Number of lanes, length, speed limits, entry/exit ramps                                     | Defines environment constraints |     

---

## 3. Model Type  

This simulation is primarily **Prescriptive**, since the goal is to design and evaluate strategies (via reinforcement learning policies) that autonomous vehicles can use to act safely and efficiently in traffic scenarios such as lane changing, merging, and overtaking.  

It also has a **Predictive aspect**, as the trained agent implicitly forecasts the outcomes of its possible actions (whether a lane change will lead to a safe maneuver or a collision) in order to maximize long-term rewards.  

It is not **Descriptive**, because the model is not focused on replicating current human driver behavior. Instead, it seeks to propose better strategies. Nor is it purely **Speculative**, since the experiments are grounded in realistic traffic environments and established simulation frameworks.  


TODO -> differents algorithms the RL ; 