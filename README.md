# 2022 COG RoboMaster Sim2Real
This repository is related to the [2022 CoG Robomaster Sim2Real](https://eval.ai/web/challenges/challenge-page/1513/overview) Challenge which is organized by CASIA DRL Team. In this repository, we will provide the implementation of our algorithm.

This code is developed with python 3.7 and the networks are trained using [PyTorch 1.10.2](https://github.com/pytorch/pytorch).

## Installation

We recommend using conda to create a virtual environment.

Step 1.In this [repository](https://github.com/DRL-CASIA/COG-sim2real-challenge), two environments (Windows/Linux) have been provided. You can get the environment file in your project.


Step 2. Create virtual environment using conda.(you must have installed anaconda/miniconda first.)

```Shell
conda env create -f environment_*.yml.

```

Step 3. Install the COG_API package with command:

```Shell
pip install CogEnvDecoder --upgrade
```

Step 4. Run the api_test.py, you will see our simulation environment.

## Usage

The experiment can be run by calling:
```Shell
python main.py
```



