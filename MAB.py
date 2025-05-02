import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.model_selection import train_test_split
from env.py import EcommerceEnv, train_df, val_df

# ===== MAB Cell =====

class EpsilonGreedyMAB:
    """
    ε-greedy Multi-Armed Bandit implementation.
    n_actions: number of arms
    epsilon: exploration probability
    """
    def __init__(self, n_actions, epsilon=0.3):
        self.n = n_actions
        self.eps = epsilon
        self.counts = np.zeros(self.n, dtype=int)
        self.values = np.zeros(self.n, dtype=float)

    def select(self):
        """Select an arm using ε-greedy."""
        if np.random.rand() < self.eps:
            return np.random.randint(self.n)
        return int(np.argmax(self.values))

    def update(self, arm, reward):
        """Update estimated value of the chosen arm with the received reward."""
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

class AlwaysArm:
    """Oracle policy: always pick the same arm."""
    def __init__(self, arm):
        self.arm = arm
    def select(self): return self.arm
    def update(self, arm, reward): pass


def run_mab(env, mab, n_episodes=32, update=True):
    """
    Run the bandit for n_episodes, returning avg cumulative reward.
    """
    returns = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            a = mab.select()
            state, r, done, _ = env.step(a)
            if update:
                mab.update(a, r)
            total_reward += r
        returns.append(total_reward)
    return float(np.mean(returns))

def plot_mab(train_rewards, val_rewards, train_regret, val_regret, window_size=10):
    """
    Plot MAB performance.
    """
    # Smooth the rewards and regrets using a moving average
    smooth = lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='valid')

    train_rewards_smooth = smooth(train_rewards)
    val_rewards_smooth = smooth(val_rewards)
    train_regret_smooth = smooth(train_regret)
    val_regret_smooth = smooth(val_regret)

    plt.figure(figsize=(12,4))

    # Rewards per epoch
    plt.subplot(1,2,1)
    plt.plot(train_rewards_smooth, label='MAB Train Reward per Epoch', marker='o')
    plt.plot(val_rewards_smooth,   label='MAB Val Reward per Epoch',   marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Episode Reward')
    plt.title('MAB Reward per Epoch')
    plt.legend()

    # Regret per epoch
    plt.subplot(1,2,2)
    plt.plot(train_regret_smooth, label='MAB Train Regret per Epoch', marker='o')
    plt.plot(val_regret_smooth,   label='MAB Val Regret per Epoch',   marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Regret (Oracle - Reward)')
    plt.title('MAB Regret per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_arm_performance(model, env, top_k=10):
    """Visualize the performance of each arm"""
    plt.figure(figsize=(15, 6))
    
    # Calculate the core metrics
    success_rates = model.values 
    pull_counts = model.counts    
    sorted_idx = np.argsort(-success_rates)[:top_k] 
    
    # Create subplots
    ax1 = plt.subplot(121)
    ax1.bar(range(top_k), success_rates[sorted_idx], color='skyblue')
    ax1.set_xticks(range(top_k))
    ax1.set_xticklabels([f"Arm {i}" for i in sorted_idx], rotation=45)
    ax1.set_ylabel("Estimated Reward (Success Rate)")
    ax1.set_title(f"Top {top_k} Arms Estimated Reward")
    
    ax2 = plt.subplot(122)
    ax2.bar(range(top_k), pull_counts[sorted_idx], color='salmon')
    ax2.set_xticks(range(top_k))
    ax2.set_xticklabels([f"Arm {i}" for i in sorted_idx], rotation=45)
    ax2.set_ylabel("Pull Count")
    ax2.set_title(f"Top {top_k} Arms Pull Count Distribution")
    
    plt.tight_layout()
    plt.show()

def train_mab(train_df, val_df, n_epochs, epsilon_start=0.5, epsilon_min=0.05, decay_rate=0.1):
    """
    Train ε-greedy MAB and compute per-epoch regret (loss) against oracle.
    Returns:
      train_rewards, val_rewards, train_regret, val_regret, mab
    """
    train_env = EcommerceEnv(train_df)
    val_env   = EcommerceEnv(val_df)
    n_arms    = train_env.action_space.n

    # Compute oracle (best-arm) performance
    oracle_train = max(
        run_mab(train_env, AlwaysArm(a), n_episodes=32, update=False)
        for a in range(n_arms)
    )
    oracle_val   = max(
        run_mab(val_env, AlwaysArm(a), n_episodes=32, update=False)
        for a in range(n_arms)
    )

    mab = EpsilonGreedyMAB(n_arms, epsilon_start)
    train_rewards, val_rewards = [], []
    train_regret,  val_regret  = [], []

    for epoch in trange(n_epochs, desc="MAB Epochs"):
        # Apply epsilon decay
        epsilon = max(epsilon_min, epsilon_start * np.exp(-decay_rate * epoch))
        mab.eps = epsilon  # Update epsilon in the MAB instance

        # Training set update and evaluation
        R_tr = run_mab(train_env, mab, update=True)
        train_rewards.append(R_tr)
        train_regret.append(oracle_train - R_tr)

        # Validation set (no update)
        R_val = run_mab(val_env, mab, update=False)
        val_rewards.append(R_val)
        val_regret.append(oracle_val - R_val)

    return train_rewards, val_rewards, train_regret, val_regret, mab  # Return the mab object

if __name__ == "__main__":

    # Run MAB training and evaluation, receiving the trained mab object
    mab_tr_R, mab_val_R, mab_tr_regret, mab_val_regret, mab = train_mab(
        train_df, val_df, n_epochs=150, epsilon_start=0.9, epsilon_min=0.02, decay_rate=0.1
    )

    # — New: print the metrics for each epoch —
    import pandas as pd
    stats = pd.DataFrame({
        'Epoch':          list(range(1, len(mab_tr_R)+1)),
        'Train Reward':   mab_tr_R,
        'Val Reward':     mab_val_R,
        'Train Regret':   mab_tr_regret,
        'Val Regret':     mab_val_regret
    })
    print("=== Metrics per Epoch ===")
    print(stats.to_string(index=False))

    # Plot MAB results
    plot_mab(mab_tr_R, mab_val_R, mab_tr_regret, mab_val_regret)

    # Plot arm performance (Top 15 arms)
    plot_arm_performance(mab, train_env, top_k=15)
