# Towards Physics-Informed Diffusion Reinforcement Learning for Investigating Adversarial Trajectories
<<<<<<< HEAD
 Implementation of AAAI 2025 Hackathon: "Towards Physics-Informed Diffusion Reinforcement Learning (PhyDRL) for Investigating Adversarial in Trajectories".

## Overview
Given trajectory data, a domain-specific study area, and a user-defined threshold, we aim to find anomalous trajectories indicative of possible deception-based activity (i.e., an activity that attempts to conceal its movements by intentionally emitting false signals). The problem, especially in the maritime domain, is societally important to curb illegal activities in international waters, such as unauthorized fishing and illicit oil transfers. The problem is computationally challenging due to a potentially infinite variety of fake trajectory behavior and a lack of ground truth to train state-of-the-art generative models. Due to data sparsity, prior approaches capture anomalous trajectories with high false positive rates, resulting in lower accuracy. To address these limitations, we propose a physics-informed diffusion model that integrates kinematic knowledge to identify trajectories that do not adhere to physical laws. Experimental results on real-world datasets in the maritime and urban domains show the proposed framework provides higher prediction accuracy than baseline generative models.
=======
 Implementation of AAAI 2025 Hackathon: "Towards Physics-Informed Diffusion Reinforcement Learning (PhyDRL) for Investigating Adversarial in Trajectories.

## Overview
PhyDRL builds upon the Denoising Diffusion Probabilistic Model (DDPM) framework to capture typical patterns in object trajectories and then flags any departures from these patterns as anomalies. In contrast to conventional diffusion approaches, it weaves physics-based constraints into the generative stage via reinforcement learning, making sure that the trajectories it synthesizes remain physically coherent and align with established motion laws. By combining a data-driven methodology with physics-informed priors, PhyDRL produces more realistic trajectory predictions that mirror real-world dynamics while naturally reducing the risk of overfitting.
>>>>>>> 2eff72eab64044018b58fb9152ca06b8c1ffd5e3

## Key Features

1. **Physics-Informed Diffusion**:  
   - Embeds domain knowledge (e.g., smooth motion constraints, kinematic equations) into the diffusion training process.  
   - Helps ensure generated trajectories adhere to real-world physics and do not exhibit abrupt or infeasible motions.

2. **Anomaly Detection**:  
   - Computes anomaly scores by comparing observed trajectories against the learned diffusion distribution.  
   - Trajectories that deviate significantly from the modeled “normal” dynamics are flagged as anomalies.

3. **Trajectory Generation**:  
   - Generate new, synthetic trajectories using the learned diffusion model.  
   - Useful for data augmentation, simulation, or planning under realistic motion constraints.

4. **Reinforcement Learning Integration** *(Optional)*:  
   - Includes a **Kinematic Bicycle Environment** (`rlPhysics-Informed-Diffusion-RL/env.py`) that can incorporate diffusion-based stochasticity for more realistic state transitions.  
   - Trains RL agents (PPO, DQN, SAC) using `train.py` or `evaluate.py` for advanced control and anomaly detection tasks in uncertain environments.


## Requirements

The required packages with python environment is:

      python>=3.7
      torch>=1.7
      pandas
      numpy
      matplotlib
      pathlib
      shutil
      datetime
      colored
      math

## Installation Instructions

1. Clone the repository: 

Download the repository to your local machine, either via git or as a ZIP download.

      git clone https://github.com/AAAI-2025-Hackathon/team_63.git
      cd PhyDRL

2. Installation Dependencies: 

The package is built with Python (>=3.8) and PyTorch. Install the required packages using pip (it is recommended to use a virtual environment):

      pip install -r requirements.txt

Otherwise, install packages individually:

      pip install torch stable-baselines3 gym pandas numpy matplotlib colored

## Running Instructions

Once the environment is set up, you can use the provided scripts to train the model, detect anomalies on new trajectories, and generate synthetic trajectories. Below are the typical usage steps:

### Training the Diffusion Model

If your primary goal is physics-informed trajectory generation or anomaly detection (without RL), run something similar to:

      python scripts/train.py \
  --data data/train_dataset.csv \
  --epochs 100 \
  --batch_size 64 \
  --out checkpoints/
                        

In this example:

--data: path to the training dataset of normal trajectories.

--epochs and --batch_size: training hyperparameters.

--out: directory for model checkpoints and logs.

This script trains the diffusion model on normal trajectory patterns, incorporating physics-based constraints (e.g., jerk minimization, velocity smoothness) from physics.py


### Detecting Anomalies

Use a trained PhyDRL model to score new trajectories:

      python scripts/detect.py \
  --model checkpoints/pidpm_model.pth \
  --data data/test_dataset.csv \
  --threshold 0.1 \
  --output results/anomalies.csv
                        

Here:

--model path to the trained PhyDRL model checkpoint (e.g., .pth).

--data file (or directory) of test trajectories.

--threshold: optional numeric threshold on the anomaly score. Trajectories with scores above this are labeled anomalous.

--output optional CSV file to save detailed anomaly results.

### Generating Synthetic Trajectories

You can generate new trajectories that mimic the learned dynamics:

```text
      python scripts/generate.py \
  --model checkpoints/pidpm_model.pth \
  --num_samples 50 \
  --out results/synthetic_trajectories.csv
```
--model: path to the trained PhyDRL checkpoint.
--num_samples: how many synthetic trajectories to generate.
--out: CSV file to store the generated trajectories.

### Reinforcement Learning Integration

We provide a Kinematic Bicycle Environment that can incorporate the diffusion model to simulate realistic state transitions, beneficial for RL training:

1. Train an RL Agent (PPO, DQN, or SAC) with diffusion-based noise:
```text
python train.py \
  --algo PPO \
  --timesteps 50000 \
  --lr 1e-3 \
  --seed 42 \
  --diffusion_model_path checkpoints/pidpm_model.pth \
  --save_path results/ppo_model
```

2. Evaluate an RL Agent:
```text
python evaluate.py \
  --algo PPO \
  --model_path results/ppo_model.zip \
  --episodes 10
```
Loads the saved RL policy and runs it in the environment to measure average rewards or other metrics.

3. Generate RL-Guided Trajectories:
```text
python generate.py \
  --algo PPO \
  --model_path results/ppo_model.zip \
  --episodes 5 \
  --max_steps 200 \
  --save_trajectories results/generated_trajectories.npy
```

4. RL-Based Anomaly Detection:
```text
python detect.py \
  --algo PPO \
  --model_path results/ppo_model.zip \
  --diffusion_model_path checkpoints/pidpm_model.pth \
  --episodes 5 \
  --max_steps 200 \
  --output_file results/anomaly_scores.csv
```

## Project Structure

```plaintext
Physics-Informed-Diffusion-RL/
│   ├── __init__.py         # Initializes the Physics-Informed-Diffusion-RL package
│   ├── diffusion.py        # Diffusion model implementation (forward & reverse process)
│   ├── physics.py          # Physics-informed constraints & losses (e.g., jerk minimization)
│   ├── models.py           # Neural net architectures for trajectory encoding/decoding
│   ├── utils.py            # Helper functions (logging, metrics, I/O)
│   ├── env.py              # (Optional) Kinematic Bicycle Environment for RL integration
├── scripts/
│   ├── train.py            # Script to train the PhyDRL model (or RL agent) on a dataset
│   ├── detect.py           # Script to compute anomaly scores for trajectories
│   ├── generate.py         # Script to generate synthetic trajectories via diffusion model
│   └── evaluate.py         # (Optional) Script to evaluate trained RL agents
├── requirements.txt        # List of required Python packages for PhyDRL
├── LICENSE                 # MIT License (or other) for this project
└── README.md               # Main project documentation (this file)                    
```
The Physics-Informed-Diffusion-RL/ directory is optional; it holds the environment and RL code. 

The Physics-Informed-Diffusion-RL/ directory remains the core library for diffusion-based anomaly detection and trajectory synthesis.

## License
This project is provided under the MIT License. See LICENSE for details.

## Contributing
Contributions are welcome! If you find bugs, have questions, or want to propose enhancements, please open an issue or submit a pull request.

## Acknowledgments
Thanks to all contributors and the open-source community for providing tools like PyTorch, Stable Baselines3, and other libraries.