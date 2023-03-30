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

The experiment can be run by calling the following code, where you may change the path to the environment file:
```Shell
python submit_test.py
```
To train a new model by calling
```Shell
python train.py
```

## Assets
In our repository, we provide demo vedio and experiment report where you can download for reference. All rights reserved. 

## Credits

This project was completed by ==Chao Li, Xiaodong Liu and Ming Sang in the University of Chinese Academy of Sciences (UCAS)==. Special thanks to my teammates for their support and help, still miss that summer when we worked together to complete this big challenge!




## Comments
Miss you guys! 🍻 ——xiaodong 





