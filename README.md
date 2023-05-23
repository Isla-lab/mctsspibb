## Implementation of MCTS-SPIBB (Scalable Safe Policy Improvement via Monte Carlo Tree Search)

This project can be used to reproduce the experiments presented in:
- Alberto Castellini, Federico Bianchi*, Edoardo Zorzi*, Thiago D. Simão, Alessandro Farinelli, Matthijs T. J. Spaan, "Scalable Safe Policy Improvement via Monte Carlo Tree Search" 

  
## Prerequisites

The project is implemented in Python 3.10, Ubuntu 20.04.5 LTS (for the full list of requirements please refer to file requirements.txt)


## Usage

We include the following:
- Libraries of the following algorithms (subfolder src):
	* SPIBB (from Romain Laroche https://github.com/RomainLaroche/SPIBB)
	* SPIBB_DP (Dynamic Programming) (our implementation)
	* MCTS-SPIBB (our implementation)

- Environments (subfolder envs):
	* Gridworld environment,
	* SysAdmin environment.
- MCTS-SPIBB experiments (see file script_running_examples.txt to learn how to set script parameters):
    * Gridworld experiment of Section 'Results on convergence'. Run:
    	`python results_on_convergence_gridworld.py`
    	
    * SysAdmin experiment of Section 'Results on convergence'. Run:
        `python results_on_convergence_sysadmin.py`
        
    * Gridworld experiment of Section 'Results on safety'. Run:
    	`python results_on_safety_gridworld.py`
    	
    * SysAdmin experiment of Section 'Results on safety'. Run:
        `python results_on_safety_sysadmin.py`

    * SysAdmin experiment of Section 'Results on scalability'. Run:
        `python results_on_scalability_sysadmin.py`
        
    * Other experiments using MCTS-SPIBB and SPIBB_DP on SysAdmin. Run:
        `python results_on_convergence_and_safety_sysadmin.py`

Each script running on domain X generates results in the folder out/X in .pkl files. In particular:
- Datasets D are saved in subfolder out/X/batch
- Mask matrices (N_D(s, a) >= N_wedge) are saved in subfolder out/X/mask
- MLE transition models are saved in subfolder out/X/mle
- Mask matrices N_D(s, a) are saved in subfolder out/X/q
- Policy performance (e.g. state-values, average discounted returns) are saved in subfolder out/X/results

Figures displayed in the paper can be generated from files in subfolder out/X/results 

## Comparison with the other state-of-the-art SPI algorithms
The comparison of MCTS-SPIBB with the other state-of-the-art SPI algorithms on the WetChicken domain was performed using the benchmark presented by:
 - P. Scholl, F. Dietrich, C. Otte, and S. Udluft. Safe Policy Improvement Approaches on Discrete Markov Decision Processes. ICAART 2022, to appear (arXiv:2201.12175)
 	(https://github.com/Philipp238/Safe-Policy-Improvement-Approaches-on-Discrete-Markov-Decision-Processes)

## License and citations

This project is GPLv3-licensed. MCTS-SPIBB authors kindly ask to cite the paper:
 - Alberto Castellini, Federico Bianchi*, Edoardo Zorzi*, Thiago D. Simão, Alessandro Farinelli, Matthijs T. J. Spaan, "Scalable Safe Policy Improvement via Monte Carlo Tree Search" if the
code is used in other projects.

