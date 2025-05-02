import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Set, DefaultDict
from datetime import datetime
from collections import defaultdict

def preprocess_data(raw_df):
    """One-stop data preprocessing function"""
    df = raw_df.copy()
    
    # Handle monetary fields
    for col in ['Sales', 'Profit', 'Shipping Cost']:
        df[col] = df[col].replace(r'[\$,]', '', regex=True).str.strip()
        df = df[df[col].str.match(r'^\d+(\.\d+)?$', na=False)]
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert order date to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df = df[~df['Order Date'].isnull()]
    
    # Encode Product ID as categorical
    df['Product ID'] = (
        df['Product Category'].astype(str) + '-' + 
        df['Product'].astype(str)
    ).astype('category').cat.codes
    
    # Encode Gender
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1}).fillna(-1)
    
    # Numeric columns
    numeric_cols = ['Age', 'Quantity', 'Discount']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
    
    # Behavior columns
    for col in ['Like', 'Share', 'Add to Cart']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df

def generate_negative_samples(
    df,
    user_col='Customer ID',
    item_col='Product ID',
    neg_ratio=1,
    max_neg_per_user=50
):
    all_users = df[user_col].unique()
    all_items = df[item_col].unique()
    item_popularity = df[item_col].value_counts()
    item_weights = 1 / (item_popularity + 1e-5)

    negative_samples = []
    seen_samples = set()

    for user in all_users:
        user_data = df[df[user_col] == user]
        purchased_items = set(user_data[item_col])
        candidate_negatives = list(set(all_items) - purchased_items)

        if not candidate_negatives:
            continue

        num_pos = len(user_data)
        max_possible = min(len(candidate_negatives), max_neg_per_user)
        num_negatives = min(num_pos * neg_ratio, max_possible)

        if len(candidate_negatives) >= num_negatives:
            negatives = random.sample(candidate_negatives, int(num_negatives))
        else:
            weights = [item_weights[item] for item in candidate_negatives]
            negatives = random.choices(candidate_negatives, weights=weights, k=int(num_negatives))

        for item in negatives:
            sample = user_data.iloc[0].copy()
            sample[item_col] = item
            sample['Add to Cart'] = 0
            sample['Like'] = 0
            sample['Browsing Time (min)'] = np.random.uniform(0, 1)
            if 'Profit' in df.columns:
                sample['Profit'] = 0

            if (user, item) not in seen_samples:
                negative_samples.append(sample.to_dict())
                seen_samples.add((user, item))

    df_neg = pd.DataFrame(negative_samples)
    df_neg = df_neg.astype(df.dtypes.to_dict())
    combined_df = pd.concat([df, df_neg], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=[user_col, item_col], keep='first')
    return combined_df

# 1. Load raw data
raw_df = pd.read_csv('E_commerce.csv', low_memory=False)
processed_df = preprocess_data(raw_df)
df_with_negatives = generate_negative_samples(processed_df, neg_ratio=1)
df_with_negatives['label'] = df_with_negatives['Add to Cart']

# 2. Discretize: Browsing Time + Quantity
df_with_negatives['browsing_bin'] = pd.qcut(
    df_with_negatives['Browsing Time (min)'],
    q=10,
    labels=False,
    duplicates='drop'
)

df_with_negatives['quantity_bin'] = pd.qcut(
    df_with_negatives['Quantity'],
    q=5,
    labels=False,
    duplicates='drop'
)

# 3. Create combined stratified label
df_with_negatives['stratify_key'] = df_with_negatives.apply(
    lambda row: f"{int(row['label'])}_{int(row['browsing_bin'])}_{int(row['quantity_bin'])}",
    axis=1
)

# 4. Shuffle data
shuffled_data = df_with_negatives.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Stratified train/validation split
train_df, val_df = train_test_split(
    shuffled_data,
    test_size=0.2,
    random_state=42,
    stratify=shuffled_data['stratify_key']
)

# 6. Drop helper columns
drop_cols = ['browsing_bin', 'quantity_bin', 'stratify_key']
train_df = train_df.drop(columns=drop_cols)
val_df = val_df.drop(columns=drop_cols)

# 7. Print split statistics
print("\nSplit results:")
print(f"Training size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Positive rate (train): {train_df['label'].mean():.3f}")
print(f"Positive rate (val): {val_df['label'].mean():.3f}")
print(f"Average browsing time (train): {train_df['Browsing Time (min)'].mean():.2f}")
print(f"Average browsing time (val): {val_df['Browsing Time (min)'].mean():.2f}")
print(f"Average quantity (train): {train_df['Quantity'].mean():.2f}")
print(f"Average quantity (val): {val_df['Quantity'].mean():.2f}")

def augment_data(df, noise_factor=0.05, scaling_factor=0.1):
    """
    Apply data augmentation techniques to expand the dataset.

    Parameters:
    - df: original DataFrame
    - noise_factor: the level of noise to add to numerical features
    - scaling_factor: the scaling factor for continuous variables like Quantity and Browsing Time

    Returns:
    - augmented_df: augmented DataFrame with new samples
    """
    augmented_data = []

    # Add noise and scaling to Browsing Time and Quantity
    for idx, row in df.iterrows():
        new_row = row.copy()
        new_row['Browsing Time (min)'] += np.random.normal(0, noise_factor)
        new_row['Quantity'] += np.random.normal(0, noise_factor)
        new_row['Browsing Time (min)'] *= (1 + np.random.uniform(-scaling_factor, scaling_factor))
        new_row['Quantity'] *= (1 + np.random.uniform(-scaling_factor, scaling_factor))
        new_row['Browsing Time (min)'] = max(new_row['Browsing Time (min)'], 0)
        new_row['Quantity'] = max(new_row['Quantity'], 0)
        augmented_data.append(new_row)

    augmented_df = pd.DataFrame(augmented_data)
    augmented_df = augmented_df.astype(df.dtypes.to_dict())
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    return combined_df

# 1. Augment the dataset
augmented_df = augment_data(df_with_negatives, noise_factor=0.05, scaling_factor=0.1)

# 2. Shuffle the dataset
shuffled_data = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Split into train and validation sets; validation is not augmented
train_df, val_df = train_test_split(
    shuffled_data,
    test_size=0.2,
    random_state=42,
    stratify=shuffled_data['stratify_key']  # Preserve class distribution
)

# 4. Print dataset statistics
print("Original dataset size:", len(df_with_negatives))
print("Augmented dataset size:", len(augmented_df))
print("Training dataset size:", len(train_df))
print("Validation dataset size:", len(val_df))


class EcommerceEnv(gym.Env):

    def __init__(
        self, 
        df: pd.DataFrame,
        history_window: int = 5,
        max_steps_per_episode: int = 100,
        regularization_factor: float = 0.005,
        stage_reward_weights: Optional[List[List[float]]] = None
    ):
        super().__init__()
        
        # ---- Data Validation and Preprocessing ----
        self._validate_data(df)
        self.df = self._preprocess_data(df.copy())
        
        # ---- Mapping Relationships ----
        self.users = self.df['Customer ID'].astype(str).unique()
        self.products = self.df['Product ID'].astype(str).unique()
        self.product_to_idx = {str(p): i for i, p in enumerate(self.products)}
        self.idx_to_product = {i: str(p) for i, p in enumerate(self.products)}
        
        # ---- Environment Parameters ----
        self.history_window = history_window
        self.max_steps = max_steps_per_episode
        self.reg_factor = regularization_factor
        self.stage_weights = stage_reward_weights or [
            [0.8, 0.1, 0.0, 0.0, 0.0],  # Attention
            [0.5, 0.3, 0.1, 0.0, 0.0],  # Interest
            [0.2, 0.2, 0.3, 0.5, 0.0],  # Desire
            [0.1, 0.1, 0.2, 0.3, 1.0]   # Action
        ]
        
        # ---- Space Definitions ----
        self.action_space = spaces.Discrete(len(self.products))
        self.observation_space = spaces.Dict({
            "user": spaces.Box(-1, 1, shape=(10,), dtype=np.float32),
            "history": spaces.MultiDiscrete([len(self.products)+1] * history_window),
            "context": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
            "aida_stage": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "stage_stats": spaces.Box(0, 1, shape=(4,), dtype=np.float32)
        })

        # ---- State Initialization ----
        self._seen_users: Set[str] = set()
        self._seen_products: Set[str] = set()
        self._popular_products = self._compute_popular_products()
        self._init_encoders()
        self.reset()

    def _validate_data(self, df: pd.DataFrame):
        """Data integrity validation"""
        required_columns = {
            'Customer ID', 'Product ID', 'Age', 'Gender',
            'Browsing Time (min)', 'Order Date', 'Profit'
        }
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data preprocessing"""
        # Type conversion
        df['Customer ID'] = df['Customer ID'].astype(str)
        df['Product ID'] = df['Product ID'].astype(str)
        
        # Feature engineering
        df['is_click'] = (df['Browsing Time (min)'] > 0).astype(int)
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['price_per_unit'] = df['Sales'] / (df['Quantity'] + 1e-6)
        
        # Handling missing values
        if 'Category' not in df.columns:
            df['Category'] = df['Product ID'].str.split('_').str[0]
            
        return df

    def _init_encoders(self):
        """Encoder initialization"""
        self.region_enc = LabelEncoder().fit(self.df['Region'].astype(str))
        self.education_enc = LabelEncoder().fit(self.df['Education'].astype(str))

    def _compute_popular_products(self, top_n: int = 100) -> Set[str]:
        """Calculate popular products (based on interaction frequency)"""
        return set(self.df['Product ID'].value_counts().head(top_n).index)

    def reset(self):
        """Environment reset (with cold start optimization)"""
        self.current_user = np.random.choice(self.users)
        
        # Cold start handling
        if self.current_user not in self._seen_users:
            self.current_aida = 0.0
            self._seen_users.add(self.current_user)
        
        self.user_data = self.df[self.df['Customer ID'] == self.current_user].reset_index(drop=True)
        self._idx = 0
        self.history = []
        self._cumulative_reward = 0.0
        self._last_info = None
        return self._get_state()

    def _get_user_features(self, row) -> np.ndarray:
        """User feature vector"""
        return np.array([
            float(row['Age']) / 100.0,
            (float(row['Gender']) + 1) / 2.0,
            self.education_enc.transform([str(row['Education'])])[0] / max(1, len(self.education_enc.classes_)),
            1.0 if str(row['Marital Status']) == 'Married' else 0.0,
            np.log1p(float(row['Sales'])) / 10.0,
            float(row['Discount']),
            float(row['Quantity']) / 10.0,
            float(str(row['Order Priority']) == 'High'),
            float(str(row['Segment']) == 'Home Office'),
            float(row['Shipping Cost']) / 20.0
        ], dtype=np.float32)

    def _get_context_features(self, row) -> np.ndarray:
        """Context features"""
        return np.array([
            row['Order Date'].hour / 24.0,
            (row['Order Date'].month - 1) / 12.0,
            self.region_enc.transform([str(row['Region'])])[0] / max(1, len(self.region_enc.classes_))
        ], dtype=np.float32)

    def _get_history(self) -> np.ndarray:
        """Historical interaction records"""
        h = [a+1 for a in self.history[-self.history_window:]]
        return h + [0] * (self.history_window - len(h))

    def _get_base_aida_stage(self, row) -> int:
        # Handle integer index case
        if isinstance(row, (int, np.integer)):
            if 0 <= row < len(self.user_data):
                row = self.user_data.iloc[row]
            else:
                return 0
        
        # Handle invalid inputs
        if not isinstance(row, pd.Series):
            return 0
        
        # Safely get behavior metrics
        try:
            current_purchase = float(row.get('Profit', 0)) > 0
            current_cart = row.get('Add to Cart', 0) > 0
            current_like = row.get('Like', 0) > 0
            browsing_time = float(row.get('Browsing Time (min)', 0))
        except (ValueError, TypeError):
            current_purchase = current_cart = current_like = False
            browsing_time = 0.0
        
        # Historical behavior analysis (only handles valid integer indices)
        recent_actions = []
        for a in self.history[-5:]:
            if isinstance(a, (int, np.integer)) and 0 <= a < len(self.user_data):
                recent_actions.append(a)
        
        # Calculate historical behaviors
        purchase_count = cart_count = share_count = long_view_count = 0
        for a in recent_actions:
            action_row = self.user_data.iloc[a]
            purchase_count += int(float(action_row.get('Profit', 0)) > 0)
            cart_count += int(action_row.get('Add to Cart', 0) > 0)
            share_count += int(action_row.get('Share', 0) > 0)
            long_view_count += int(float(action_row.get('Browsing Time (min)', 0)) > 2)
        
        # Stage determination (priority from high to low)
        if current_purchase or purchase_count > 0:
            return 3  # Action
        elif current_cart or cart_count > 0 or share_count > 0:
            return 2  # Desire
        elif long_view_count >= 2 or current_like or browsing_time > 2:
            return 1  # Interest
        return 0  # Attention

    def _get_dynamic_aida_stage(self, row) -> float:
        """Dynamic stage calculation (with time decay)"""
        base_stage = self._get_base_aida_stage(row)
        if isinstance(row, pd.Series):
            time_decay = np.exp(-(datetime.now() - row['Order Date']).days / 60)  # 60-day half-life
        else:
            time_decay = 1.0
            
        transition_weight = min(0.5, len(self.history)*0.1)
        smooth_stage = (1-transition_weight)*self.current_aida + transition_weight*(base_stage/3.0)
        
        if isinstance(row, pd.Series) and row['Product ID'] not in self._seen_products:
            self._seen_products.add(row['Product ID'])
            return max(0.2, smooth_stage)
        
        return min(1.0, smooth_stage * time_decay)

    def _get_state(self) -> Dict[str, np.ndarray]:
        """Enhanced state representation"""
        row = self.user_data.iloc[self._idx] if self._idx < len(self.user_data) else self.df.iloc[0]
        self.current_aida = self._get_dynamic_aida_stage(row)
        
        # Stage statistics
        stage_counts = np.zeros(4)
        for a in self.history[-self.history_window:]:
            stage = min(3, int(self._get_base_aida_stage(a) if a < len(self.user_data) else 0))
            stage_counts[stage] += 1
        stage_stats = stage_counts / len(self.history) if self.history else np.zeros(4)
        
        return {
            "user": self._get_user_features(row),
            "history": self._get_history(),
            "context": self._get_context_features(row),
            "aida_stage": np.array([self.current_aida], dtype=np.float32),
            "stage_stats": stage_stats.astype(np.float32)
        }

    def step(self, action: int) -> tuple:
        """Execute action"""
        product = str(self.idx_to_product[action])
        row = self.user_data.iloc[self._idx] if self._idx < len(self.user_data) else None
        
        # Initialize info
        info = {
            "product": product,
            "valid_interaction": False,
            "aida_stage": float(self.current_aida),
            "stage_transition": None,
            "reward_components": {
                "base": 0.0,
                "diversity": 0.0,
                "regularization": 0.0
            }
        }

        # 1. Calculate base reward
        prev_stage = self.current_aida
        if self.current_aida < 0.3 and product not in self._popular_products:
            reward = -0.01
        elif row is None:
            reward = 0.0
        elif str(row['Product ID']) == product:
            reward = self._calculate_enhanced_reward(row)
            info["valid_interaction"] = True
        else:  # Partial match
            reward = 0.15
            if 'Category' in row and row['Category'] == product.split('_')[0]:
                reward += 0.05
        info["reward_components"]["base"] = reward

        # 2. Stage update
        current_row = row if row is not None else self.df.iloc[0]
        new_stage = self._get_dynamic_aida_stage(current_row)
        info["stage_transition"] = {
            "from": float(prev_stage),
            "to": float(new_stage),
            "reason": self._get_transition_reason(current_row, prev_stage, new_stage)
        }
        self.current_aida = new_stage

        # 3. Reward composition
        diversity = self._diversity_bonus()
        reg_term = min(0.05, self.reg_factor * self._get_regularization_term())
        
        final_reward = reward + diversity - reg_term
        info["reward_components"].update({
            "diversity": diversity,
            "regularization": -reg_term,
            "final": final_reward
        })
        
        # 4. State update
        self.history.append(action)
        self._idx += 1
        done = (self._idx >= len(self.user_data)) or (self._idx >= self.max_steps)
        self._cumulative_reward += final_reward
        self._last_info = info
        
        return self._get_state(), final_reward, done, info

    def _calculate_enhanced_reward(self, row: pd.Series) -> float:
        """Standardized reward calculation"""
        metrics = {
            "browsing_time": float(row['Browsing Time (min)']),
            "like": int(row.get('Like', 0) > 0),
            "share": int(row.get('Share', 0) > 0),
            "cart": int(row.get('Add to Cart', 0) > 0),
            "purchased": int(float(row['Profit']) > 0)
        }
        
        stage = min(3, int(self.current_aida * 3.99))
        weights = self.stage_weights[stage]
        
        # Behavior rewards
        reward = weights[0] * np.tanh(metrics["browsing_time"] / 5)  # Browsing time
        reward += weights[1] * metrics["like"]  # Like
        reward += weights[2] * metrics["share"] # Share
        reward += weights[3] * metrics["cart"]  # Add to cart
        
        # Purchase reward (with upper limit)
        purchase_reward = min(weights[4] * metrics["purchased"], 6.0)
        reward += purchase_reward
        
        # Decay factor
        decay = 1.0 / (1.0 + 0.1 * len(self.history))
        return decay * max(0.05, reward)

    def _diversity_bonus(self) -> float:
        """Diversity reward"""
        if len(self.history) < 3:
            return 0.0
        last_5 = self.history[-5:]
        return 0.1 * (len(set(last_5)) / len(last_5))

    def _get_regularization_term(self) -> float:
        """Regularization term (based on average browsing time)"""
        idxs = [i for i in self.history[-self.history_window:] if i < len(self.user_data)]
        if not idxs:
            return 0
        avg_time = np.mean([float(self.user_data.iloc[i]['Browsing Time (min)']) for i in idxs])
        return avg_time / 10

    def _get_transition_reason(self, row, prev: float, curr: float) -> str:
        """Stage transition reason"""
        if curr > prev:
            if curr >= 0.75: return "purchase_detected"
            elif curr >= 0.5: return "cart_or_share" 
            else: return "engagement_achieved"
        return "natural_decay"

    def render(self, mode='human'):
        """Visualize state"""
        if mode == 'human' and self._last_info:
            print(f"\nUser: {self.current_user} | Step: {self._idx}")
            print(f"Stage: {self.current_aida:.2f} | Reward: {self._cumulative_reward:.2f}")
            print(f"Last Action: {self._last_info['product']}")
            print(f"Transition: {self._last_info['stage_transition']['reason']}")

    def close(self):
        """Clean up resources"""
        self._seen_users.clear()
        self._seen_products.clear()
