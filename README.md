# CODACAN+HTransR

## Overview
This repository contains the implementation of the framework proposed in the paper:

**"Distributional Actor Critic with Hierarchical Attention-based State Representation for Dialogue Policy Learning"**  
Authors: Yasaman Saffari, Javad Salimi Sartakhti  
Affiliation: Department of Electrical and Computer Engineering, University of Kashan, Kashan, Iran  
Contact: [saffari@kashanu.ac.ir](mailto:saffari@kashanu.ac.ir), [salimi@kashanu.ac.ir](mailto:salimi@kashanu.ac.ir)

---

## Abstract
Dialogue management plays a critical role in dialogue systems, comprising a dialogue state tracker and a policy learner. This project introduces a novel framework that integrates:
- **Hierarchical Attention-based State Representation (HTransR):** Captures intra- and inter-state dependencies in dialogue state sequences using a static transformer-based model, TransCAT, and a dual attention mechanism for effective state aggregation.
- **Distributional Actor-Critic Algorithm with Context-Aware Noise (CODACAN):** A reinforcement learning algorithm that combines a dueling network architecture with context-aware Gaussian noise for robust dialogue policy learning.

---

## Features
1. **CODACAN Algorithm:** 
   - Actor-Critic model with distributional dueling architecture.
   - Context-aware noise for action distribution learning.

2. **HTransR Framework:** 
   - Hierarchical attention mechanism for sequential state representation.
   - Static Transformer architecture (TransCAT) adapted for multi-hot state inputs in reinforcement learning tasks.

3. **Experimental Validation:**
   - Evaluated on a real-world dataset collected from Amazon Mechanical Turk.
   - Outperforms baseline models in terms of automatic evaluation metrics.

---

## Repository Structure
- `deep_dialog/`: Contains the source code for the framework.
- `data/`: Includes the preprocessed movie ticket booking, restaurant reservation, and taxi ordering dataset.
- `script/`: Scripts for running experiments and generating results.
- `README.md`: This file.
- `finalhit.ipynb`: Run codes for train in colab.

---

## Requirements
To reproduce the experiments, install the following dependencies:
- Python >= 3.8
- PyTorch >= 1.10
- Transformers >= 4.0
- NumPy, Scikit-learn, and Matplotlib

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yasamansaffarii/CODACAN-HTransR.git
   cd CODACAN-HTransR
   ```

2. Train the DQN Agent:
   - **Movie Ticket Booking domain:**  
     ```bash
     bash script/movie.sh
     ```
   - **Restaurant Reservation domain:**  
     ```bash
     bash script/restaurant.sh
     ```
   - **Taxi Ordering domain:**  
     ```bash
     bash script/taxi.sh
     ```

---

## Hyperparameter Settings
1. Set domain-specific parameters:  
   Modify the following lines in `deep_dialog/qlearning/network.py` (Lines 408-411):  
   ```python
   X = mov if domain = movie
   X = res if domain = restaurant
   X = tax if domain = taxi
   self.state_element_sizes = self.state_element_sizes_X
   self.state_lens = self.state_len_X
   self.pad = self.pad_X
   self.max_size = self.max_size_X
   ```

2. Adjust the sequence state length:  
   - `deep_dialog/qlearning/dist_dqn.py` (Line 27)  
   - `deep_dialog/agents/agent_dqn.py` (Line 32)  
   ```python
   self.seq_len = 3 or 5 or 7
   ```

   - For only the current state, execute:  
     ```bash
     cd CODACAN+HTransR0
     ```

---

## Citation
If you use this work, please cite:

@article{Saffari2025CODACAN,  
  title={Distributional Actor Critic with Hierarchical Attention-based State Representation for Dialogue Policy Learning},  
  author={Yasaman Saffari and Javad Salimi Sartakhti},  
  journal={Under Review},  
  year={2025},  
  organization={University of Kashan}  
}

---

## License
This repository is currently restricted and cannot be forked, modified, or redistributed.

---

## References
This repository is based on the code repositories TC-Bot and DialogDQN-Variants, as well as the paper *End-to-End Task-Completion Neural Dialogue Systems and Dialogue Environments are Different from Games: Investigating Variants of Deep Q-Networks for Dialogue Policy*. This repo is a modified version of the referenced repositories, performing at a similar level of accuracy for baseline models, although direct comparability is not established.



## Contact
For questions or collaborations, feel free to reach out to:

Yasaman Saffari: saffari@kashanu.ac.ir  
Javad Salimi Sartakhti: salimi@kashanu.ac.ir

