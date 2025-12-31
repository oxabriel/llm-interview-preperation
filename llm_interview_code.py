"""
大模型算法岗面试 - 手撕代码合集
包含 Transformer、PPO、DPO、GRPO 等核心算法实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ============================================================================
# 第一部分：Transformer 核心组件
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (batch, n_heads, seq_len, d_k)
            k: (batch, n_heads, seq_len, d_k)
            v: (batch, n_heads, seq_len, d_v)
            mask: (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len)
        Returns:
            output: (batch, n_heads, seq_len, d_v)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        d_k = q.size(-1)
        # (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len)
        """
        batch_size = q.size(0)
        
        # Linear projections and split into heads
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attn_weights = self.attention(q, k, v, mask)
        
        # Concatenate heads
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        x = self.W_o(x)
        x = self.dropout(x)
        
        return x, attn_weights


class PositionalEncoding(nn.Module):
    """绝对位置编码（正弦余弦）"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class RotaryPositionalEmbedding(nn.Module):
    """RoPE 旋转位置编码（用于 LLaMA 等模型）"""
    def __init__(self, dim, max_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len
    
    def forward(self, x, seq_len=None):
        """
        Args:
            x: (batch, seq_len, n_heads, d_k)
        Returns:
            cos, sin for rotary embedding
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    """SwiGLU 激活函数（现代 LLM 常用）"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        SwiGLU(x) = (Swish(xW1) ⊙ xW3)W2
        """
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    """RMS Normalization（LLaMA 使用）"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim)
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder层"""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len)
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder层（带Cross-Attention）"""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: (batch, tgt_seq_len, d_model)
            enc_output: (batch, src_seq_len, d_model)
            src_mask: (batch, 1, 1, src_seq_len)
            tgt_mask: (batch, 1, tgt_seq_len, tgt_seq_len) - causal mask
        """
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x


def create_causal_mask(seq_len, device='cpu'):
    """创建因果掩码（用于 GPT 类模型）"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def create_padding_mask(seq, pad_idx=0):
    """创建 padding 掩码"""
    # seq: (batch, seq_len)
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    return mask


# ============================================================================
# 第二部分：KV Cache 实现
# ============================================================================

class KVCache:
    """KV Cache for efficient autoregressive generation"""
    def __init__(self):
        self.k_cache = None
        self.v_cache = None
    
    def update(self, k, v):
        """
        Args:
            k, v: (batch, n_heads, new_seq_len, d_k)
        Returns:
            updated k, v: (batch, n_heads, total_seq_len, d_k)
        """
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        
        return self.k_cache, self.v_cache
    
    def clear(self):
        self.k_cache = None
        self.v_cache = None


# ============================================================================
# 第三部分：强化学习算法 - PPO
# ============================================================================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算 Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: (batch, seq_len) - 每步的奖励
        values: (batch, seq_len) - 每步的价值估计
        dones: (batch, seq_len) - 是否结束
        gamma: 折扣因子
        lam: GAE lambda
    
    Returns:
        advantages: (batch, seq_len)
        returns: (batch, seq_len)
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]
        
        # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
        
        # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
        advantages[:, t] = last_gae = delta + gamma * lam * (1 - dones[:, t]) * last_gae
    
    returns = advantages + values
    return advantages, returns


def ppo_loss(log_probs, old_log_probs, advantages, values, returns, 
             clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
    """
    PPO 损失函数
    
    Args:
        log_probs: (batch, seq_len) - 新策略的log概率
        old_log_probs: (batch, seq_len) - 旧策略的log概率
        advantages: (batch, seq_len) - 优势函数
        values: (batch, seq_len) - 价值估计
        returns: (batch, seq_len) - 实际回报
        clip_epsilon: PPO clip 范围
        value_coef: 价值损失系数
        entropy_coef: 熵正则化系数
    
    Returns:
        total_loss, policy_loss, value_loss, entropy
    """
    # 计算概率比 ratio = π_new / π_old
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss (MSE)
    value_loss = F.mse_loss(values, returns)
    
    # Entropy bonus (假设已计算)
    entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
    
    # Total loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    
    return total_loss, policy_loss, value_loss, entropy


class PPOTrainer:
    """PPO 训练器示例"""
    def __init__(self, policy_model, value_model, optimizer, 
                 gamma=0.99, lam=0.95, clip_epsilon=0.2,
                 value_coef=0.5, entropy_coef=0.01, 
                 ppo_epochs=4, batch_size=64):
        self.policy_model = policy_model
        self.value_model = value_model
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
    
    def train_step(self, states, actions, rewards, dones, old_log_probs):
        """
        单步训练
        
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len)
            rewards: (batch, seq_len)
            dones: (batch, seq_len)
            old_log_probs: (batch, seq_len)
        """
        # Compute values
        with torch.no_grad():
            values = self.value_model(states).squeeze(-1)
        
        # Compute advantages and returns
        advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.lam)
        
        # PPO update for multiple epochs
        for _ in range(self.ppo_epochs):
            # Get new log probs and values
            logits = self.policy_model(states)
            new_log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = new_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            new_values = self.value_model(states).squeeze(-1)
            
            # Compute loss
            loss, policy_loss, value_loss, entropy = ppo_loss(
                new_log_probs, old_log_probs, advantages, new_values, returns,
                self.clip_epsilon, self.value_coef, self.entropy_coef
            )
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_model.parameters()) + list(self.value_model.parameters()),
                max_norm=0.5
            )
            self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()


# ============================================================================
# 第四部分：DPO (Direct Preference Optimization)
# ============================================================================

def dpo_loss(policy_logps_chosen, policy_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """
    DPO 损失函数
    
    Args:
        policy_logps_chosen: (batch,) - 策略模型对chosen的log概率
        policy_logps_rejected: (batch,) - 策略模型对rejected的log概率
        ref_logps_chosen: (batch,) - 参考模型对chosen的log概率
        ref_logps_rejected: (batch,) - 参考模型对rejected的log概率
        beta: KL散度的温度参数
    
    Returns:
        loss, rewards_chosen, rewards_rejected
    
    DPO目标: maximize log σ(β * log(π(y_w|x) / π_ref(y_w|x)) - β * log(π(y_l|x) / π_ref(y_l|x)))
    其中 y_w 是 chosen, y_l 是 rejected
    """
    # Compute implicit rewards
    rewards_chosen = beta * (policy_logps_chosen - ref_logps_chosen)
    rewards_rejected = beta * (policy_logps_rejected - ref_logps_rejected)
    
    # DPO loss (negative log sigmoid)
    logits = rewards_chosen - rewards_rejected
    loss = -F.logsigmoid(logits).mean()
    
    return loss, rewards_chosen.mean(), rewards_rejected.mean()


def compute_sequence_logps(model, input_ids, labels, attention_mask=None):
    """
    计算序列的log概率（用于DPO）
    
    Args:
        model: 语言模型
        input_ids: (batch, seq_len)
        labels: (batch, seq_len) - 下一个token的标签
        attention_mask: (batch, seq_len)
    
    Returns:
        log_probs: (batch,) - 每个序列的平均log概率
    """
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    
    # Shift for causal LM: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Compute log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask padding tokens and average
    if attention_mask is not None:
        mask = attention_mask[:, 1:].contiguous()
        gathered_log_probs = gathered_log_probs * mask
        seq_log_probs = gathered_log_probs.sum(dim=1) / mask.sum(dim=1)
    else:
        seq_log_probs = gathered_log_probs.mean(dim=1)
    
    return seq_log_probs


class DPOTrainer:
    """DPO 训练器"""
    def __init__(self, policy_model, ref_model, optimizer, beta=0.1):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.beta = beta
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def train_step(self, chosen_input_ids, chosen_labels, chosen_attention_mask,
                   rejected_input_ids, rejected_labels, rejected_attention_mask):
        """
        单步DPO训练
        
        Args:
            chosen_*: chosen样本的输入
            rejected_*: rejected样本的输入
        """
        # Compute log probs for policy model
        policy_logps_chosen = compute_sequence_logps(
            self.policy_model, chosen_input_ids, chosen_labels, chosen_attention_mask
        )
        policy_logps_rejected = compute_sequence_logps(
            self.policy_model, rejected_input_ids, rejected_labels, rejected_attention_mask
        )
        
        # Compute log probs for reference model
        with torch.no_grad():
            ref_logps_chosen = compute_sequence_logps(
                self.ref_model, chosen_input_ids, chosen_labels, chosen_attention_mask
            )
            ref_logps_rejected = compute_sequence_logps(
                self.ref_model, rejected_input_ids, rejected_labels, rejected_attention_mask
            )
        
        # Compute DPO loss
        loss, rewards_chosen, rewards_rejected = dpo_loss(
            policy_logps_chosen, policy_logps_rejected,
            ref_logps_chosen, ref_logps_rejected, self.beta
        )
        
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), rewards_chosen.item(), rewards_rejected.item()


# ============================================================================
# 第五部分：GRPO (Group Relative Policy Optimization)
# ============================================================================

def grpo_loss(log_probs, old_log_probs, rewards, group_size=4, 
              kl_coef=0.1, clip_epsilon=0.2):
    """
    GRPO 损失函数
    
    Args:
        log_probs: (batch, seq_len) - 新策略log概率
        old_log_probs: (batch, seq_len) - 旧策略log概率
        rewards: (batch,) - 每个样本的奖励
        group_size: 组大小（每组内样本数）
        kl_coef: KL散度惩罚系数
        clip_epsilon: clip范围
    
    Returns:
        loss
    """
    batch_size = log_probs.size(0)
    assert batch_size % group_size == 0
    
    # Reshape into groups: (num_groups, group_size, seq_len)
    num_groups = batch_size // group_size
    log_probs_grouped = log_probs.view(num_groups, group_size, -1)
    old_log_probs_grouped = old_log_probs.view(num_groups, group_size, -1)
    rewards_grouped = rewards.view(num_groups, group_size)
    
    # Compute advantages as deviation from group mean
    group_mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
    group_std_rewards = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
    advantages = (rewards_grouped - group_mean_rewards) / group_std_rewards
    
    # Flatten back
    advantages = advantages.view(-1, 1)
    
    # Compute ratio
    log_probs_sum = log_probs.sum(dim=-1, keepdim=True)
    old_log_probs_sum = old_log_probs.sum(dim=-1, keepdim=True)
    ratio = torch.exp(log_probs_sum - old_log_probs_sum)
    
    # Clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # KL penalty
    kl_div = (old_log_probs_sum - log_probs_sum).mean()
    
    loss = policy_loss + kl_coef * kl_div
    
    return loss


# ============================================================================
# 第六部分：Reward Model
# ============================================================================

class RewardModel(nn.Module):
    """奖励模型（用于RLHF）"""
    def __init__(self, base_model, hidden_dim):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            rewards: (batch,)
        """
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
        
        # Use last token's hidden state
        if attention_mask is not None:
            last_indices = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_indices]
        else:
            last_hidden = hidden_states[:, -1, :]
        
        rewards = self.reward_head(last_hidden).squeeze(-1)
        return rewards


def reward_model_loss(rewards_chosen, rewards_rejected, margin=0.0):
    """
    奖励模型的Pairwise Ranking Loss
    
    Args:
        rewards_chosen: (batch,)
        rewards_rejected: (batch,)
        margin: 最小margin
    
    Returns:
        loss
    """
    # Loss: -log(σ(r_chosen - r_rejected - margin))
    loss = -F.logsigmoid(rewards_chosen - rewards_rejected - margin).mean()
    return loss


# ============================================================================
# 第七部分：优化器和学习率调度
# ============================================================================

class AdamW:
    """AdamW 优化器实现"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # Initialize momentum and velocity
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters with weight decay (AdamW style)
            param.data = param.data * (1 - self.lr * self.weight_decay)
            param.data = param.data - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
    """
    带warmup的余弦学习率调度
    
    Args:
        optimizer: 优化器
        num_warmup_steps: warmup步数
        num_training_steps: 总训练步数
        min_lr: 最小学习率
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 第八部分：采样和解码策略
# ============================================================================

def top_k_sampling(logits, k=50, temperature=1.0):
    """
    Top-K 采样
    
    Args:
        logits: (batch, vocab_size)
        k: 保留top-k个最高概率的token
        temperature: 温度参数
    
    Returns:
        sampled_token: (batch,)
    """
    # Apply temperature
    logits = logits / temperature
    
    # Get top-k values and indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # Create a mask for non-top-k tokens
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
    
    # Sample from the filtered distribution
    probs = F.softmax(mask, dim=-1)
    sampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled_token


def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Top-P (Nucleus) 采样
    
    Args:
        logits: (batch, vocab_size)
        p: 累积概率阈值
        temperature: 温度参数
    
    Returns:
        sampled_token: (batch,)
    """
    # Apply temperature
    logits = logits / temperature
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Keep at least one token
    sorted_indices_to_remove[..., 0] = False
    
    # Set removed tokens to -inf
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # Unsort and sample
    logits_filtered = torch.zeros_like(logits).scatter_(
        dim=-1, index=sorted_indices, src=sorted_logits
    )
    probs_filtered = F.softmax(logits_filtered, dim=-1)
    sampled_token = torch.multinomial(probs_filtered, num_samples=1).squeeze(-1)
    
    return sampled_token


def beam_search(model, input_ids, max_length=50, num_beams=5, eos_token_id=2):
    """
    Beam Search 解码
    
    Args:
        model: 语言模型
        input_ids: (batch, seq_len) - 初始输入
        max_length: 最大生成长度
        num_beams: beam大小
        eos_token_id: 结束token的ID
    
    Returns:
        generated_ids: (batch, max_length)
    """
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # Initialize beams: (batch * num_beams, seq_len)
    input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(batch_size * num_beams, -1)
    
    # Beam scores: (batch, num_beams)
    beam_scores = torch.zeros(batch_size, num_beams, device=device)
    beam_scores[:, 1:] = float('-inf')
    beam_scores = beam_scores.view(-1)
    
    # Track finished sequences
    done = [False] * batch_size
    
    for _ in range(max_length):
        # Get model outputs
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        next_token_logits = logits[:, -1, :]
        
        # Compute log probabilities
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # Add to beam scores
        next_scores = beam_scores.unsqueeze(-1) + log_probs  # (batch * num_beams, vocab_size)
        
        # Reshape for beam selection
        next_scores = next_scores.view(batch_size, num_beams * log_probs.size(-1))
        
        # Select top beams
        next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1, largest=True, sorted=True)
        
        # Convert to beam indices and token indices
        next_beam_indices = next_tokens // log_probs.size(-1)
        next_tokens = next_tokens % log_probs.size(-1)
        
        # Prepare next input
        beam_indices = next_beam_indices.view(-1)
        input_ids = input_ids[beam_indices]
        input_ids = torch.cat([input_ids, next_tokens.view(-1, 1)], dim=-1)
        
        beam_scores = next_scores.view(-1)
        
        # Check for EOS
        if (next_tokens == eos_token_id).any():
            break
    
    # Return best beam for each batch
    return input_ids.view(batch_size, num_beams, -1)[:, 0, :]


# ============================================================================
# 第九部分：评估指标
# ============================================================================

def compute_perplexity(model, input_ids, labels, attention_mask=None):
    """
    计算困惑度 (Perplexity)
    
    Args:
        model: 语言模型
        input_ids: (batch, seq_len)
        labels: (batch, seq_len)
        attention_mask: (batch, seq_len)
    
    Returns:
        perplexity: float
    """
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        perplexity = torch.exp(loss).item()
    
    return perplexity


def compute_bleu(references, hypotheses, n=4):
    """
    简化版 BLEU Score 计算
    
    Args:
        references: List[List[str]] - 参考句子（已分词）
        hypotheses: List[str] - 生成句子（已分词）
        n: n-gram 最大值
    
    Returns:
        bleu_score: float
    """
    from collections import Counter
    
    def get_ngrams(tokens, n):
        """获取n-gram"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def modified_precision(ref, hyp, n):
        """计算修正精确度"""
        hyp_ngrams = Counter(get_ngrams(hyp, n))
        ref_ngrams = Counter(get_ngrams(ref, n))
        
        # Clip counts
        clipped_counts = {ngram: min(count, ref_ngrams[ngram]) 
                         for ngram, count in hyp_ngrams.items()}
        
        numerator = sum(clipped_counts.values())
        denominator = max(1, sum(hyp_ngrams.values()))
        
        return numerator / denominator
    
    # Compute precision for each n-gram
    precisions = []
    for i in range(1, n+1):
        prec_sum = sum(modified_precision(ref, hyp, i) 
                      for ref, hyp in zip(references, hypotheses))
        precisions.append(prec_sum / len(hypotheses))
    
    # Geometric mean
    if min(precisions) > 0:
        log_precisions = [math.log(p) for p in precisions]
        geo_mean = math.exp(sum(log_precisions) / n)
    else:
        geo_mean = 0
    
    # Brevity penalty
    ref_len = sum(len(ref) for ref in references)
    hyp_len = sum(len(hyp) for hyp in hypotheses)
    
    if hyp_len > ref_len:
        bp = 1
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0
    
    bleu_score = bp * geo_mean
    return bleu_score


# ============================================================================
# 第十部分：LoRA (Low-Rank Adaptation)
# ============================================================================

class LoRALayer(nn.Module):
    """
    LoRA: Low-Rank Adaptation of Large Language Models
    在预训练权重上添加低秩分解矩阵进行微调
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices: W = W_0 + (alpha/rank) * B * A
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x, base_output):
        """
        Args:
            x: (batch, seq_len, in_features)
            base_output: (batch, seq_len, out_features) - 来自冻结的预训练层
        
        Returns:
            output: (batch, seq_len, out_features)
        """
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        return base_output + lora_output * self.scaling


class LinearWithLoRA(nn.Module):
    """带LoRA的线性层"""
    def __init__(self, base_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.base_layer = base_layer
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Add LoRA
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
    
    def forward(self, x):
        base_output = self.base_layer(x)
        return self.lora(x, base_output)


# ============================================================================
# 第十一部分：Gradient Checkpointing
# ============================================================================

def gradient_checkpointing_forward(module, *args):
    """
    Gradient Checkpointing: 用计算换内存
    在前向传播时不保存中间激活，反向传播时重新计算
    
    Args:
        module: 要应用checkpointing的模块
        args: 模块的输入
    
    Returns:
        output: 模块的输出
    """
    def custom_forward(*inputs):
        return module(*inputs)
    
    return torch.utils.checkpoint.checkpoint(custom_forward, *args)


# ============================================================================
# 第十二部分：混合精度训练
# ============================================================================

class MixedPrecisionTrainer:
    """混合精度训练包装器"""
    def __init__(self, model, optimizer, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler if scaler else torch.cuda.amp.GradScaler()
    
    def train_step(self, inputs, targets):
        """
        混合精度训练步骤
        
        Args:
            inputs: 输入数据
            targets: 目标标签
        """
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping (在unscale之后)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()


# ============================================================================
# 第十三部分：Tokenization - BPE
# ============================================================================

def get_stats(vocab):
    """计算词汇表中相邻token pair的频率"""
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def merge_vocab(pair, vocab):
    """合并词汇表中的token pair"""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    
    return new_vocab


def train_bpe(text, num_merges=100):
    """
    训练 BPE (Byte Pair Encoding)
    
    Args:
        text: str - 训练文本
        num_merges: int - BPE合并次数
    
    Returns:
        merges: List[Tuple] - BPE合并规则
    """
    # Initialize vocabulary with character-level tokens
    words = text.split()
    vocab = {}
    for word in words:
        word = ' '.join(list(word)) + ' </w>'
        vocab[word] = vocab.get(word, 0) + 1
    
    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        
        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
        
        print(f"Merge {i+1}: {best_pair}")
    
    return merges


# ============================================================================
# 示例使用代码
# ============================================================================

def example_transformer():
    """Transformer 使用示例"""
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create attention mask (causal mask for GPT)
    mask = create_causal_mask(seq_len)
    
    # Create encoder layer
    encoder = TransformerEncoderLayer(d_model, n_heads)
    output = encoder(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def example_ppo():
    """PPO 使用示例"""
    batch_size = 32
    seq_len = 128
    state_dim = 512
    action_dim = 50000  # vocab size
    
    # Mock models
    policy_model = lambda x: torch.randn(batch_size, seq_len, action_dim)
    value_model = lambda x: torch.randn(batch_size, seq_len, 1)
    
    # Mock data
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    rewards = torch.randn(batch_size, seq_len)
    dones = torch.zeros(batch_size, seq_len)
    
    # Compute old log probs
    logits = policy_model(states)
    old_log_probs = F.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    # Compute values
    values = value_model(states).squeeze(-1)
    
    # Compute GAE
    advantages, returns = compute_gae(rewards, values, dones)
    
    print(f"Advantages shape: {advantages.shape}")
    print(f"Returns shape: {returns.shape}")


def example_dpo():
    """DPO 使用示例"""
    batch_size = 8
    
    # Mock log probabilities
    policy_logps_chosen = torch.randn(batch_size)
    policy_logps_rejected = torch.randn(batch_size) - 1.0  # Lower probs for rejected
    ref_logps_chosen = torch.randn(batch_size)
    ref_logps_rejected = torch.randn(batch_size)
    
    # Compute DPO loss
    loss, rewards_chosen, rewards_rejected = dpo_loss(
        policy_logps_chosen, policy_logps_rejected,
        ref_logps_chosen, ref_logps_rejected, beta=0.1
    )
    
    print(f"DPO Loss: {loss.item():.4f}")
    print(f"Rewards Chosen: {rewards_chosen.item():.4f}")
    print(f"Rewards Rejected: {rewards_rejected.item():.4f}")


if __name__ == "__main__":
    print("=" * 80)
    print("大模型算法面试 - 手撕代码示例")
    print("=" * 80)
    
    print("\n1. Transformer Example:")
    example_transformer()
    
    print("\n2. PPO Example:")
    example_ppo()
    
    print("\n3. DPO Example:")
    example_dpo()
    
    print("\n所有核心算法实现完成！")
    print("建议按以下顺序复习：")
    print("1. Transformer基础架构（Attention, FFN, LayerNorm）")
    print("2. 位置编码（Sinusoidal, RoPE）")
    print("3. PPO算法（GAE, Clip Objective）")
    print("4. DPO算法（Implicit Reward, Preference Learning）")
    print("5. GRPO算法（Group-based Advantage）")
    print("6. 优化技巧（LoRA, Gradient Checkpointing, Mixed Precision）")