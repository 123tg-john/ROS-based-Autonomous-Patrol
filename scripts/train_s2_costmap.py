#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# --- 核心模組與函式庫引入 ---
import rospy
import os
import time
import numpy as np
import random
import rospkg
import pickle
import glob
import sys
from collections import OrderedDict, deque 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 引入 ROS 標準訊息格式，用於發佈 KPI
from std_msgs.msg import Float32

# 引入自定義的 ROS 模擬環境與基礎 SAC 演算法元件
from turtlebot3_sac_env_stage2 import TurtleBot3EnvStage3
from train_sac import ReplayBuffer, weights_init_, soft_update, hard_update, SAC as SAC_Base

# ===============================================================================================
# === 1. 神經網路架構定義 (Neural Network Architectures) ===
# ===============================================================================================
# SAC 演算法由「演員 (Actor)」和「評論家 (Critic)」兩種類型的網路組成。

class PolicyNetworkStage3(nn.Module):
    """
    策略網路 (演員 Actor)，負責制定決策。
    這個客製化的版本將輸出分為兩部分：一部分用於駕駛，另一部分用於異常判斷。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetworkStage3, self).__init__()
        # 共用的特徵提取層
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 駕駛動作的決策頭 (輸出動作的平均值和標準差的對數)
        self.driving_mean_linear = nn.Linear(hidden_dim, action_dim - 1)
        self.driving_log_std_linear = nn.Linear(hidden_dim, action_dim - 1)
        
        # 異常分數的決策頭 (輸出一個 0-1 之間的值)
        self.anomaly_score_linear = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        driving_mean = self.driving_mean_linear(x)
        driving_log_std = self.driving_log_std_linear(x)
        driving_log_std = torch.clamp(driving_log_std, min=-20, max=2) # 限制標準差範圍以穩定訓練
        
        # 使用 Sigmoid 函式將輸出壓縮到 0-1 區間，作為異常分數
        anomaly_score = torch.sigmoid(self.anomaly_score_linear(x))
        
        return driving_mean, driving_log_std, anomaly_score

    def sample(self, state):
        """ 從策略中採樣一個具體的動作 """
        driving_mean, driving_log_std, anomaly_score = self.forward(state)
        driving_std = driving_log_std.exp()
        normal = torch.distributions.Normal(driving_mean, driving_std)
        
        driving_x_t = normal.rsample() # 重參數化技巧，允許梯度回傳
        driving_action = torch.tanh(driving_x_t) # 將駕駛動作壓縮到 -1 到 1
        
        # 將駕駛動作和異常分數合併成最終的完整動作向量
        action = torch.cat([driving_action, anomaly_score], dim=1)
        
        # 計算熵，這是 SAC 演算法的核心之一
        log_prob = normal.log_prob(driving_x_t)
        log_prob -= torch.log(1 - driving_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class QNetworkStage3(nn.Module):
    """
    Q網路 (評論家 Critic)，負責評估 Actor 決策的好壞。
    採用 Twin-Q 設計（兩個Q網路），以緩解 Q 值高估問題，提升訓練穩定性。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetworkStage3, self).__init__()
        # 第一個 Q 網路
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, 1)
        # 第二個 Q 網路
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1) # 將狀態和動作拼接作為輸入
        x1 = F.relu(self.linear1_q1(xu)); x1 = F.relu(self.linear2_q1(x1)); x1 = self.linear3_q1(x1)
        x2 = F.relu(self.linear1_q2(xu)); x2 = F.relu(self.linear2_q2(x2)); x2 = self.linear3_q2(x2)
        return x1, x2

# ===============================================================================================
# === 2. SAC Agent 主體定義 ===
# ===============================================================================================
# 繼承自基礎的 SAC 類別，並替換為我們客製化的神經網路

class SAC_Stage3(SAC_Base):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(SAC_Stage3, self).__init__(state_dim, action_dim, **kwargs)
        
        self.lr = kwargs.get('lr', 0.00005)

        self.policy = PolicyNetworkStage3(state_dim, action_dim).to(self.device)
        self.critic = QNetworkStage3(state_dim, action_dim).to(self.device)
        self.critic_target = QNetworkStage3(state_dim, action_dim).to(self.device)
        
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, eval=False):
        """ 根據當前狀態選擇一個動作 """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        """ 從經驗池中採樣數據，更新所有神經網路的權重 """
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target
        
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        self.q_loss = qf_loss.item()
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_loss = policy_loss.item()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_loss = alpha_loss.item()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        soft_update(self.critic_target, self.critic, self.tau)

    def load_checkpoint_for_stage3_transfer(self, stage1_episode, models_path):
        """ 特殊函式：實現從第一階段模型的遷移學習 """
        stage1_checkpoint_path = os.path.join(models_path, f"checkpoint_ep{stage1_episode}.pth")
        if not os.path.exists(stage1_checkpoint_path):
            rospy.logerr(f"錯誤：找不到用於遷移的第一階段檢查點檔案 (Ep {stage1_episode})。")
            return False
        
        rospy.loginfo(f"--- 開始從第一階段 (Ep {stage1_episode}) 進行遷移學習 ---")
        s1_checkpoint = torch.load(stage1_checkpoint_path, map_location=self.device)
        
        s3_policy_dict = self.policy.state_dict()
        s1_policy_dict = s1_checkpoint['policy_state_dict']
        
        for name, param in s1_policy_dict.items():
            if name in s3_policy_dict and s3_policy_dict[name].shape == param.shape:
                s3_policy_dict[name].copy_(param)
                rospy.loginfo(f"  [Policy] 成功遷移層: {name}")
        
        self.policy.load_state_dict(s3_policy_dict)

        s3_critic_dict = self.critic.state_dict()
        s1_critic_dict = s1_checkpoint['critic_state_dict']

        for name, param in s1_critic_dict.items():
            if name in s3_critic_dict and s3_critic_dict[name].shape == param.shape:
                s3_critic_dict[name].copy_(param)
                rospy.loginfo(f"  [Critic] 成功遷移層: {name}")

        self.critic.load_state_dict(s3_critic_dict)
        hard_update(self.critic_target, self.critic)
        rospy.loginfo("*** 成功將第一階段的駕駛技巧遷移至新模型！***")
        return True

# ===============================================================================================
# === 3. 主訓練程序 (Main Training Program) ===
# ===============================================================================================
if __name__ == '__main__':
    rospy.init_node('sac_train_stage3')
    env = TurtleBot3EnvStage3()
    
    # --- 使用者設定 ---
    TRANSFER_FROM_STAGE1_EPISODE = 7355
    FORCE_LOAD_EPISODE = None 
    
    # --- 訓練超參數設定 ---
    max_episodes = 20001
    max_steps = 500
    replay_buffer_size = 100000
    BATCH_SIZE = 1024
    RANDOM_STEPS_WARMUP = 2000
    
    HIGH_SCORE_THRESHOLD = 0.8  # 分數高於此值，視為「明確判斷為異常」
    LOW_SCORE_THRESHOLD = 0.2   # 分數低於此值，視為「明確判斷為安全」
    
    # 學習率排程參數
    initial_lr = 0.0001
    final_lr = 0.00001
    decay_episodes = 15000

    agent = SAC_Stage3(env.state_dim, env.action_dim, lr=initial_lr) 
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # --- 路徑與日誌設定 ---
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('blender_map_project')
    s1_models_path = os.path.join(pkg_path, "models_checkpoints")
    s3_models_path = os.path.join(pkg_path, "models_checkpoints_stage3")
    run_log_dir = os.path.join(pkg_path, "tensorboard_logs_stage3", 'run_' + time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(s3_models_path): os.makedirs(s3_models_path)
    if not os.path.exists(run_log_dir): os.makedirs(run_log_dir)
    writer = SummaryWriter(run_log_dir)
    
    # --- KPI 追蹤器與 ROS 發佈器 ---
    recent_outcomes = deque(maxlen=100)
    pub_success_rate = rospy.Publisher('/patrol_kpi/success_rate', Float32, queue_size=10)
    pub_completion_time = rospy.Publisher('/patrol_kpi/completion_time', Float32, queue_size=10)
    pub_motion_smoothness = rospy.Publisher('/patrol_kpi/motion_smoothness', Float32, queue_size=10)

    start_episode = 0
    skip_warmup = False

    # --- 智慧啟動邏輯 ---
    rospy.loginfo("--- 檢查第二階段訓練進度 ---")
    latest_episode_found = None
    if os.path.exists(s3_models_path):
        checkpoints = glob.glob(os.path.join(s3_models_path, "checkpoint_ep*.pth"))
        if checkpoints:
            latest_episode_found = max([int(f.split('ep')[1].split('.pth')[0]) for f in checkpoints])

    if FORCE_LOAD_EPISODE is not None:
        rospy.loginfo(f"--- 強制從 Ep {FORCE_LOAD_EPISODE} 恢復第二階段訓練 ---")
        load_success, loaded_buffer = agent.load_checkpoint(FORCE_LOAD_EPISODE, s3_models_path)
        if load_success:
            start_episode = FORCE_LOAD_EPISODE + 1
            if loaded_buffer: replay_buffer = loaded_buffer
            skip_warmup = True
        else:
            rospy.logerr("強制載入失敗，程式終止。"); exit()
    elif latest_episode_found is not None:
        rospy.loginfo(f"--- 偵測到第二階段最新進度為 Ep {latest_episode_found}，將從此繼續 ---")
        load_success, loaded_buffer = agent.load_checkpoint(latest_episode_found, s3_models_path)
        if load_success:
            start_episode = latest_episode_found + 1
            if loaded_buffer: replay_buffer = loaded_buffer
            skip_warmup = True
        else:
            rospy.logerr(f"載入 Ep {latest_episode_found} 失敗，程式終止。"); exit()
    else:
        rospy.loginfo(f"--- 未找到第二階段進度，將從第一階段 (Ep {TRANSFER_FROM_STAGE1_EPISODE}) 進行首次遷移 ---")
        if agent.load_checkpoint_for_stage3_transfer(TRANSFER_FROM_STAGE1_EPISODE, s1_models_path):
            start_episode = 0
            skip_warmup = True
        else:
            rospy.logerr("遷移學習失敗，程式終止。"); exit()

    rospy.loginfo(f'--- 開始第二階段訓練 (從 Episode {start_episode} 開始) ---')
    
    total_steps = 0
    try:
        # --- 主訓練迴圈 ---
        for ep in range(start_episode, max_episodes):
            
            # 學習率排程
            if ep < decay_episodes:
                new_lr = initial_lr - (initial_lr - final_lr) * (ep / decay_episodes)
            else:
                new_lr = final_lr
            for param_group in agent.policy_optim.param_groups:
                param_group['lr'] = new_lr
            for param_group in agent.critic_optim.param_groups:
                param_group['lr'] = new_lr
            writer.add_scalar('hyperparameters/learning_rate', new_lr, ep)

            state = env.reset()
            if state is None:
                rospy.logwarn(f"Episode {ep} 因環境重置失敗而被跳過。")
                continue
            
            # 初始化回合數據記錄器
            ep_reward, ep_drive_reward, ep_detect_reward, ep_anomaly_score = 0, 0, 0, []
            ep_start_time = time.time()
            ep_angular_jerks = [] 
            last_angular_vel_action = 0.0
            
            high_score_count = 0
            low_score_count = 0
            
            # --- 單一回合內的步驟迴圈 ---
            for step in range(max_steps):
                if not skip_warmup and len(replay_buffer) < RANDOM_STEPS_WARMUP:
                    action = np.random.uniform(low=-1.0, high=1.0, size=env.action_dim)
                    action[2] = np.random.uniform(low=0.0, high=1.0)
                else:
                    action = agent.select_action(state)
                    action[:2] = np.clip(action[:2] + np.random.normal(0, 0.1, size=2), -1.0, 1.0)
                
                current_angular_vel_action = action[1]
                angular_jerk = abs(current_angular_vel_action - last_angular_vel_action)
                ep_angular_jerks.append(angular_jerk)
                last_angular_vel_action = current_angular_vel_action

                next_state, reward, done, info = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
                
                if len(replay_buffer) > BATCH_SIZE:
                    agent.update_parameters(replay_buffer, BATCH_SIZE)
                
                state = next_state
                ep_reward += reward
                ep_drive_reward += info.get('driving_reward', 0)
                ep_detect_reward += info.get('detection_reward', 0)
                
                anomaly_score = info.get('anomaly_score', 0.5)
                ep_anomaly_score.append(anomaly_score)
                total_steps += 1

                if anomaly_score > HIGH_SCORE_THRESHOLD:
                    high_score_count += 1
                elif anomaly_score < LOW_SCORE_THRESHOLD:
                    low_score_count += 1

                if done or info.get('is_collision', False):
                    break
            
            # --- 回合結束，計算與發佈 KPIs ---
            is_success = done and not info.get('is_collision', False)
            recent_outcomes.append(1 if is_success else 0)
            current_success_rate = sum(recent_outcomes) / len(recent_outcomes) if len(recent_outcomes) > 0 else 0.0
            pub_success_rate.publish(Float32(current_success_rate))

            if is_success:
                completion_time = time.time() - ep_start_time
                pub_completion_time.publish(Float32(completion_time))
                writer.add_scalar('kpi/task_completion_time', completion_time, ep)

            if ep_angular_jerks:
                avg_jerk = np.mean(ep_angular_jerks)
                motion_smoothness = 1.0 / (1.0 + avg_jerk) 
                pub_motion_smoothness.publish(Float32(motion_smoothness))
                writer.add_scalar('kpi/motion_smoothness', motion_smoothness, ep)

            # --- TensorBoard 記錄 ---
            writer.add_scalar('reward/total_reward', ep_reward, ep)
            writer.add_scalar('reward/driving_reward_part', ep_drive_reward, ep)
            writer.add_scalar('reward/detection_reward_part', ep_detect_reward, ep)
            writer.add_scalar('loss/q_loss', agent.q_loss, ep)
            writer.add_scalar('loss/policy_loss', agent.policy_loss, ep)
            writer.add_scalar('loss/alpha_loss', agent.alpha_loss, ep)
            writer.add_scalar('kpi/success_rate_moving_avg_100', current_success_rate, ep)
            if ep_anomaly_score:
                writer.add_scalar('metrics/avg_anomaly_score', np.mean(ep_anomaly_score), ep)
                writer.add_scalar('metrics/high_score_count', high_score_count, ep)
                writer.add_scalar('metrics/low_score_count', low_score_count, ep)

            avg_score_this_ep = np.mean(ep_anomaly_score) if ep_anomaly_score else 0.0
            rospy.loginfo(f"Ep: {ep}, Reward: {ep_reward:.2f}, Steps: {step+1}, AvgScore: {avg_score_this_ep:.2f}, "
                          f"High: {high_score_count}, Low: {low_score_count}, "
                          f"LR: {new_lr:.3e}, SuccessRate (last 100): {current_success_rate:.2%}")
            
            rospy.loginfo("")
            
            # --- 定期儲存與重啟邏輯 ---
            is_restart_episode = (ep % 180 == 0 and ep > 0)
            is_save_episode = (ep % 100 == 0 and ep > 0)

            if is_restart_episode or is_save_episode:
                if is_restart_episode:
                    rospy.loginfo(f"\n{'='*60}\n達到第 {ep} 回合的計畫性重啟點，儲存最後的檢查點...\n{'='*60}")
                agent.save_checkpoint(ep, replay_buffer, s3_models_path)
        
            if is_restart_episode:
                rospy.loginfo("檢查點儲存完畢。程式將正常退出，由外部腳本自動重啟。")
                sys.exit(0)

    except KeyboardInterrupt:
        rospy.loginfo("\n使用者手動中斷訓練...")
    finally:
        rospy.loginfo("執行最終儲存...")
        final_episode = ep if 'ep' in locals() else start_episode
        agent.save_checkpoint(final_episode, replay_buffer, s3_models_path)
        writer.close()
        rospy.loginfo("最終進度儲存完畢。程式結束。")


