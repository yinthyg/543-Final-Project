# 543-Final-Project

## Abstract
This project aims to design and implement a reinforcement learning-based recommender system to provide personalized recommendation suggestions. The model will be trained on  the UserBehavior dataset from Taobao, and we will compare three RL-based recommendation models: multi-armed bandits (MAB), contextual bandits (CB), and deep RL models (DQN, Double DQN, Multi-Step DQN). 

## Method & implementation
This project will be using Taobao User Behavior Dataset E_commerce.csv. This dataset contains detailed information about Taobao platform users and their shopping behaviors, suitable for e-commerce analysis scenarios such as customer behavior analysis, market segmentation, and sales prediction. The dataset uses "Customer ID" as the unique identifier, recording users' basic attributes, order information, and interaction behaviors. 

## Files
1. E-Commerce.csv: dataset
2. Env.py: enviroment
3. MAB.py: multi-armed Bandits with visualizations
4. CB.py: contextual bandits model with visualizations
5. DQN.py: DQN models and visualizations

## Start
To get started, clone the repository and install the required dependencies.
dependencies:
```bash
pip install numpy matplotlib tqdm scikit-learn pandas

### Run the Environment
# Before running any models, **please run `Env.py` first** to preprocess the dataset and initialize the environment:
python Env.py
