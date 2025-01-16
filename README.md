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
   - Hierarchical attention mechanism for sequencial state representation.
   - Static Transformer architecture (TransCAT) adapted for multi-hot state inputs in reinforcement learning tasks.

3. **Experimental Validation:**
   - Evaluated on a real-world dataset collected from Amazon Mechanical Turk.
   - Outperforms baseline models in terms of automatic evaluation metrics.

---

## Repository Structure
- `src/`: Contains the source code for the framework.
- `data/`: Includes the preprocessed movie ticket booking, restaurant reservation, taxi ordering dataset.
- `experiments/`: Scripts for running experiments and generating results.
- `README.md`: This file.

---

## Requirements
To reproduce the experiments, install the following dependencies:
- Python >= 3.8
- PyTorch >= 1.10
- Transformers >= 4.0
- NumPy, Scikit-learn, and Matplotlib

## How to Run
- Clone the repository:
  git clone https://github.com/yasamansaffarii/CODACAN-HTransR.git
  cd CODACAN-HTransR
  
-Train DQN Agent

  Movie-Ticket Booking domain: bash script/movie.sh
  
  Restaurant Reservation domain: bash script/restaurant.sh
  
  Taxi Ordering domain: bash script/taxi.sh

## Hyperparmeter setting
  -Set diferent domain:
    deep_dialog/qlearning/network.py---->LINE 408-411 : X=mov if domain=movie, X=res if domain restaurant X=tax if domain taxi
              self.state_element_sizes = self.state_element_sizes_X
              self.state_lens = self.state_len_X
              self.pad = self.pad_X
              self.max_size = self.max_size_X
  - Set different sequence of state length:
      deep_dialog/qlearning/dist_dqn.py---> LINE 27
      deep_dialog/agents/agent_dqn.py---> LINE 32
          self.seq_len = 3 or 5 or 7
      For only current state:
          cd CODACAN+HTransR0

## Citation
If you use this work, please cite:

@article{Saffari2025CODACAN,
  title={Distributional Actor Critic with Hierarchical Attention-based State Representation for Dialogue Policy Learning},
  author={Yasaman Saffari and Javad Salimi Sartakhti},
  journal={Under Review},
  year={2025},
  organization={University of Kashan}
}

License
This repository is currently restricted and cannot be forked, modified, or redistributed.

Contact
For questions or collaborations, feel free to reach out to:

Yasaman Saffari: saffari@kashanu.ac.ir
Javad Salimi Sartakhti: salimi@kashanu.ac.ir



