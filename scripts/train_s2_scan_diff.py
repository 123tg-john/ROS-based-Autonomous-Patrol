#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# --- 核心模組與函式庫引入 ---
import rospy
import os
import time
import numpy as np
import random
import rospkg
import glob
import sys
from collections import deque # 引入 deque，一種高效的雙端佇列，用於計算滑動平均

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # 用於記錄訓練數據以在 TensorBoard 中視覺化

from std_msgs.msg import Float32 # ROS 標準訊息格式，用於發佈 KPI

from turtlebot3_sac_env_stage3 import TurtleBot3EnvStage3 # 引入我們客製化的 ROS 訓練環境
from train_sac import ReplayBuffer, weights_init_, soft_update, hard_update, SAC as SAC_Base # 引入基礎元件

# ===============================================================================================
# === 1. 神經網路架構定義 (Neural Network Architectures) ===
# ===============================================================================================

class PolicyNetworkStage3(nn.Module):
    """
    策略網路 (演員 Actor)，負責制定決策。
    這個客製化的版本將輸出分為兩部分：一部分用於駕駛，另一部分用於異常判斷。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetworkStage3, self).__init__()
        # 共用的特徵提取層，負責從 56 維的狀態中提取有用的特徵
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 駕駛動作的決策頭，輸出動作分佈的平均值和標準差的對數
        self.driving_mean_linear = nn.Linear(hidden_dim, action_dim - 1)
        self.driving_log_std_linear = nn.Linear(hidden_dim, action_dim - 1)
        
        # 異常分數的決策頭，輸出一個單一的值，後續會通過 sigmoid 轉換
        self.anomaly_score_linear = nn.Linear(hidden_dim, 1)
        
        # 初始化網路權重
        self.apply(weights_init_)

        # 公平的初始判斷，讓異常分數的初始輸出接近 0.5 ★★★
        # 這是透過將輸出層的權重和偏置初始化為接近零的極小值來實現的。
        # 這樣，送入 sigmoid 函數前的值會接近 0，而 sigmoid(0) 的結果正好是 0.5。
        self.anomaly_score_linear.weight.data.uniform_(-3e-4, 3e-4)
        self.anomaly_score_linear.bias.data.uniform_(-3e-4, 3e-4)

    def forward(self, state):
        """定義神經網路的前向傳播路徑。"""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        # 計算駕駛動作的平均值和標準差
        driving_mean = self.driving_mean_linear(x)
        driving_log_std = self.driving_log_std_linear(x)
        driving_log_std = torch.clamp(driving_log_std, min=-20, max=2) # 限制標準差範圍以穩定訓練
        
        # 使用 Sigmoid 函式將輸出壓縮到 0-1 區間，作為異常分數
        anomaly_score = torch.sigmoid(self.anomaly_score_linear(x))
        
        return driving_mean, driving_log_std, anomaly_score

    def sample(self, state):
        """從策略中採樣一個具體的、帶有隨機性的動作，並計算其對數機率。"""
        driving_mean, driving_log_std, anomaly_score = self.forward(state)
        driving_std = driving_log_std.exp() # 從 log_std 計算出標準差
        normal = torch.distributions.Normal(driving_mean, driving_std) # 建立一個常態分佈
        
        # 使用重參數化技巧 (rsample) 進行採樣，允許梯度回傳
        driving_x_t = normal.rsample()
        # 使用 tanh 函數將採樣值壓縮到 [-1, 1] 範圍，作為最終的駕駛動作
        driving_action = torch.tanh(driving_x_t)
        
        # 將駕駛動作和異常分數合併成最終的完整動作向量
        action = torch.cat([driving_action, anomaly_score], dim=1)
        
        # 計算這個採樣動作的對數機率 (log_prob)，這是 SAC 演算法更新時的關鍵
        log_prob = normal.log_prob(driving_x_t)
        log_prob -= torch.log(1 - driving_action.pow(2) + 1e-6) # 根據 tanh 進行修正
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class QNetworkStage3(nn.Module):
    """
    Q網路 (評論家 Critic)，負責評估「在某個狀態下，採取某個動作」的好壞 (Q-value)。
    採用 Twin-Q 設計（兩個Q網路），以緩解 Q 值高估問題，提升訓練穩定性。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetworkStage3, self).__init__()
        # 第一個 Q 網路
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, 1)
        # 第二個 Q 網路 (結構相同但權重獨立)
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        """定義 Q 網路的前向傳播，同時輸出兩個 Q 網路的評分。"""
        xu = torch.cat([state, action], 1) # 將狀態和動作拼接作為輸入
        # 計算 Q1
        x1 = F.relu(self.linear1_q1(xu)); x1 = F.relu(self.linear2_q1(x1)); x1 = self.linear3_q1(x1)
        # 計算 Q2
        x2 = F.relu(self.linear1_q2(xu)); x2 = F.relu(self.linear2_q2(x2)); x2 = self.linear3_q2(x2)
        return x1, x2

# ===============================================================================================
# === 2. SAC Agent 主體定義 ===
# ===============================================================================================

class SAC_Stage3(SAC_Base):
    """繼承自基礎的 SAC 類別，並替換為我們客製化的神經網路。"""
    def __init__(self, state_dim, action_dim, **kwargs):
        super(SAC_Stage3, self).__init__(state_dim, action_dim, **kwargs)
        self.lr = kwargs.get('lr', 0.00005)

        # 實例化客製化的網路
        self.policy = PolicyNetworkStage3(state_dim, action_dim).to(self.device)
        self.critic = QNetworkStage3(state_dim, action_dim).to(self.device)
        self.critic_target = QNetworkStage3(state_dim, action_dim).to(self.device)
        
        # 為新網路建立對應的優化器
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # 硬更新目標網路，確保初始權重一致
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, eval=False):
        """根據當前狀態，從策略網路中採樣一個動作，用於與環境互動。"""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        """
        SAC 演算法的核心學習步驟。
        從經驗池中採樣一批數據，並用來更新 Actor 和 Critic 兩個網路的權重。
        """
        # 從經驗池 (Replay Buffer) 中隨機抽取一批過去的經驗
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)
        
        # 將數據轉換為 PyTorch Tensor，並移動到指定的計算設備 (CPU 或 GPU)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # --- 1. 更新評論家 (Critic) 網路 ---
        with torch.no_grad(): # 目標值的計算不進行梯度追蹤，以增加穩定性
            # 從策略網路中採樣下一個狀態的動作
            next_state_action, next_state_log_pi = self.policy.sample(next_state_batch)
            # 使用目標 Q 網路計算下一個狀態的 Q 值
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            # 選擇兩個目標 Q 值中較小的一個，以抑制 Q 值高估
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # 計算目標 Q 值 (Bellman 方程)
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target
        
        # 計算當前 Q 網路的預測值
        qf1, qf2 = self.critic(state_batch, action_batch)
        # 計算兩個 Q 網路的均方誤差損失 (MSE Loss)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        self.q_loss = qf_loss.item() # 記錄損失值，用於監控
        # 反向傳播並更新 Critic 網路的權重
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # --- 2. 更新演員 (Actor) 網路 ---
        pi, log_pi = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # 策略損失的目標是最大化 Q 值和熵
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_loss = policy_loss.item() # 記錄損失值
        # 反向傳播並更新 Actor 網路的權重
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # --- 3. 更新 Alpha (熵係數) ---
        # SAC 會自動調整 alpha，以平衡探索和利用
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_loss = alpha_loss.item() # 記錄損失值
        # 反向傳播並更新 alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # --- 4. 軟更新目標網路 ---
        # 緩慢地將主網路的權重複製到目標網路，以增加訓練穩定性
        soft_update(self.critic_target, self.critic, self.tau)

    def load_checkpoint_for_stage3_transfer(self, stage1_episode, models_path):
        """特殊函式：實現從第一階段模型的遷移學習，只載入匹配的層。"""
        stage1_checkpoint_path = os.path.join(models_path, f"checkpoint_ep{stage1_episode}.pth")
        if not os.path.exists(stage1_checkpoint_path):
            print(f"錯誤：找不到用於遷移的第一階段檢查點檔案 (Ep {stage1_episode})。")
            return False
        print(f"--- 開始從第一階段 (Ep {stage1_episode}) 進行遷移學習 ---")
        s1_checkpoint = torch.load(stage1_checkpoint_path, map_location=self.device)
        
        s3_policy_dict, s1_policy_dict = self.policy.state_dict(), s1_checkpoint['policy_state_dict']
        for name, param in s1_policy_dict.items():
            if name in s3_policy_dict and s3_policy_dict[name].shape == param.shape:
                s3_policy_dict[name].copy_(param)
                print(f"  [Policy] 成功遷移層: {name}")
        self.policy.load_state_dict(s3_policy_dict)
        
        s3_critic_dict, s1_critic_dict = self.critic.state_dict(), s1_checkpoint['critic_state_dict']
        for name, param in s1_critic_dict.items():
            if name in s3_critic_dict and s3_critic_dict[name].shape == param.shape:
                s3_critic_dict[name].copy_(param)
                print(f"  [Critic] 成功遷移層: {name}")
        self.critic.load_state_dict(s3_critic_dict)
        hard_update(self.critic_target, self.critic)
        print("*** 成功將第一階段的駕駛技巧遷移至新模型！***")
        return True

# ===============================================================================================
# === 3. 主訓練程序 (Main Training Program) ===
# ===============================================================================================
if __name__ == '__main__':
    rospy.init_node('sac_train_stage3')
    
    # --- 使用者設定 ---
    TRANSFER_FROM_STAGE1_EPISODE = 7355 # 如果是首次訓練，要從哪個第一階段模型遷移
    FORCE_LOAD_EPISODE = None # 手動指定要從哪個第三階段檢查點恢復，設為 None 則自動尋找最新
    
    # --- 訓練超參數設定 ---
    max_episodes = 30001
    max_steps = 500
    replay_buffer_size = 100000
    BATCH_SIZE = 1024
    RANDOM_STEPS_WARMUP = 2000
    
    # --- 兩階段學習率排程設定 ---
    STABILIZATION_PERIOD = 5000 # 階段一：固定學習率的持續回合數
    STABILIZATION_LR = 5e-5     # 階段一：使用的固定學習率
    initial_lr = 0.0001         # 階段二：線性衰減的起始學習率
    final_lr = 1e-6             # 階段二：線性衰減的最終學習率
    decay_episodes = 25000      # 階段二：衰減的總週期
    
    # --- 路徑設定 ---
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('blender_map_project')
    s1_models_path = os.path.join(pkg_path, "models_checkpoints")
    s3_models_path = os.path.join(pkg_path, "models_checkpoints_stage3")
    run_log_dir = os.path.join(pkg_path, "tensorboard_logs_stage3", 'run_' + time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(s3_models_path): os.makedirs(s3_models_path)
    if not os.path.exists(run_log_dir): os.makedirs(run_log_dir)

    # 先確定起始回合數，再建立環境和 Agent  ---
    # 這是為了解決環境和訓練腳本計數器不同步的問題
    start_episode = 0
    # 步驟 1: 檢查是否有舊進度，確定 start_episode
    print("--- 檢查第三階段訓練進度 ---")
    latest_episode_found = None
    if os.path.exists(s3_models_path):
        checkpoints = glob.glob(os.path.join(s3_models_path, "checkpoint_ep*.pth"))
        if checkpoints:
            latest_episode_found = max([int(f.split('ep')[1].split('.pth')[0]) for f in checkpoints])

    if FORCE_LOAD_EPISODE is not None:
        start_episode = FORCE_LOAD_EPISODE + 1
    elif latest_episode_found is not None:
        start_episode = latest_episode_found + 1
    else:
        start_episode = 0

    # 步驟 2: 建立環境，並將正確的起始回合數傳遞進去
    env = TurtleBot3EnvStage3(start_episode=start_episode)
    
    # 步驟 3: 建立 Agent, Replay Buffer 和 TensorBoard Writer
    agent = SAC_Stage3(env.state_dim, env.action_dim, lr=STABILIZATION_LR) 
    replay_buffer = ReplayBuffer(replay_buffer_size)
    writer = SummaryWriter(run_log_dir)
    
    # --- 智慧啟動邏輯 ---
    skip_warmup = False
    if start_episode > 0: # 只有在斷點續訓時才載入模型
        load_ep = start_episode - 1
        print(f"--- 從 Ep {load_ep} 恢復第三階段訓練 ---")
        load_success, loaded_buffer = agent.load_checkpoint(load_ep, s3_models_path)
        if load_success:
            if loaded_buffer: replay_buffer = loaded_buffer
            skip_warmup = True
        else:
            rospy.logerr(f"載入 Ep {load_ep} 失敗，程式終止。"); exit()
    else: # 從頭開始訓練時，執行遷移學習
        print(f"--- 未找到第三階段進度，將從第一階段 (Ep {TRANSFER_FROM_STAGE1_EPISODE}) 進行首次遷移 ---")
        if agent.load_checkpoint_for_stage3_transfer(TRANSFER_FROM_STAGE1_EPISODE, s1_models_path):
            skip_warmup = True
        else:
            rospy.logerr("遷移學習失敗，程式終止。"); exit()

    print(f'--- 開始第三階段訓練 (從 Episode {start_episode} 開始) ---')
    
    try:
        # --- 主訓練迴圈 ---
        for ep in range(start_episode, max_episodes):
            
            # --- 兩階段學習率排程邏輯 ---
            # 計算穩定期的結束回合點
            stabilization_end_episode = start_episode + STABILIZATION_PERIOD if STABILIZATION_PERIOD else start_episode
            if ep < stabilization_end_episode:
                # 階段一：穩定適應期，使用固定的學習率
                new_lr = STABILIZATION_LR
            else:
                # 階段二：精細微調期，恢復線性衰減算法
                if ep < decay_episodes:
                    progress = ep / decay_episodes
                    new_lr = initial_lr - (initial_lr - final_lr) * progress
                else:
                    new_lr = final_lr # 衰減結束後，保持在最低學習率

            # 將計算出的新學習率應用到兩個優化器
            for param_group in agent.policy_optim.param_groups: param_group['lr'] = new_lr
            for param_group in agent.critic_optim.param_groups: param_group['lr'] = new_lr
            writer.add_scalar('hyperparameters/learning_rate', new_lr, ep)

            # --- 單一回合開始 ---
            state = env.reset()
            if state is None:
                print(f"Episode {ep} 因環境重置失敗而被跳過。"); continue
            
            ep_reward, ep_anomaly_score = 0, []
            
            # --- 單一回合內的步驟迴圈 ---
            for step in range(max_steps):
                # 熱身階段，採取隨機動作以增加探索
                if not skip_warmup and len(replay_buffer) < RANDOM_STEPS_WARMUP:
                    action = np.random.uniform(low=-1.0, high=1.0, size=env.action_dim)
                    action[2] = np.random.uniform(low=0.0, high=1.0)
                else:
                    # Agent 根據當前狀態選擇動作
                    action = agent.select_action(state)
                    # 為駕駛動作增加少量探索噪聲，有助於跳出局部最優
                    action[:2] = np.clip(action[:2] + np.random.normal(0, 0.1, size=2), -1.0, 1.0)
                
                # Agent 與環境互動，獲得回饋
                next_state, reward, done, info = env.step(action)
                # 將這次的經驗 (s, a, r, s', done) 存入經驗池
                replay_buffer.push(state, action, reward, next_state, done)
                
                # 如果經驗池中的數據足夠，就進行一次學習
                if len(replay_buffer) > BATCH_SIZE:
                    agent.update_parameters(replay_buffer, BATCH_SIZE)
                
                # 更新狀態和記錄
                state = next_state
                ep_reward += reward
                ep_anomaly_score.append(info.get('anomaly_score', 0))
                
                # 如果回合結束 (撞牆或跑滿500步)，則跳出步驟迴圈
                if done or info.get('is_collision', False): break
            
            # --- 回合結束，記錄數據 ---
            avg_score_this_ep = np.mean(ep_anomaly_score) if ep_anomaly_score else 0.0
            print(f"Ep: {ep}, Reward: {ep_reward:.2f}, Steps: {step+1}, AvgScore: {avg_score_this_ep:.2f}, LR: {new_lr:.3e}")
            
            # 將各種指標寫入 TensorBoard
            writer.add_scalar('reward/total_reward', ep_reward, ep)
            writer.add_scalar('loss/q_loss', agent.q_loss, ep)
            writer.add_scalar('loss/policy_loss', agent.policy_loss, ep)
            if ep_anomaly_score:
                writer.add_scalar('metrics/avg_anomaly_score', avg_score_this_ep, ep)

            # 每 100 回合儲存一次檢查點
            if (ep % 100 == 0 and ep > 0):
                agent.save_checkpoint(ep, replay_buffer, s3_models_path)

    except KeyboardInterrupt:
        print("\n使用者手動中斷訓練...")
    finally:
        # 無論是正常結束還是手動中斷，都執行最終儲存
        print("執行最終儲存...")
        final_episode = ep if 'ep' in locals() else start_episode
        agent.save_checkpoint(final_episode, replay_buffer, s3_models_path)
        writer.close()
        print("最終進度儲存完畢。程式結束。")


