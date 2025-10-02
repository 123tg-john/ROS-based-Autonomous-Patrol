#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================================
巡邏任務指揮中心 (v3.0 - Topic輸出版)
================================================================================================
- ★★★ 核心架構修改 (ROS 原生方案) ★★★: 根據使用者建議，回歸至標準的終端機日誌模式。
                                        - 移除所有 curses 介面程式碼，恢復原生滾動日誌。
                                        - 新增一個 ROS 發佈者 (Publisher)，將即時的「異常分數」
                                          發佈至 /patrol/anomaly_score 主題。
                                        - 搭配 `rostopic echo` 指令，即可在獨立終端機中
                                          監看分數，實現了邏輯與顯示的完美解耦。
- 沿用 v1.6 的穩定執行緒輸入模型。
- 雙模型架構與核心導航邏輯保持不變。
"""

import rospy
import rospkg
import os
import yaml
import numpy as np
import math
import time
import glob
import random
import threading
import tf
import sys

# ROS 訊息與服務
from std_msgs.msg import Float32 # (+) 新增：用於發佈異常分數
from geometry_msgs.msg import Twist, Pose, Quaternion, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from gazebo_msgs.srv import SetModelState, SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState

# PyTorch (用於載入和執行SAC模型)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 從訓練腳本中引入兩個不同的網路架構
from train_sac_stage2 import PolicyNetworkStage3, weights_init_

# 為第一階段的導航模型定義一個獨立、匹配的網路架構
class PolicyNetworkStage1(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetworkStage1, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim) 
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state)); x = F.relu(self.linear2(x))
        mean = self.mean_linear(x); log_std = self.log_std_linear(x)
        return mean, torch.clamp(log_std, min=-20, max=2)

    def sample(self, state):
        mean, log_std = self.forward(state); std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample(); action = torch.tanh(x_t)
        return action, None

class PatrolManager:
    def __init__(self):
        rospy.init_node('patrol_manager_ros')
        rospy.loginfo("--- 巡邏任務指揮中心 (v3.0 - Topic輸出版) 初始化 ---")

        # 初始化所有屬性
        self.robot_pose = Pose()
        self.robot_velocity = Twist()
        self.static_map_data = None
        self.static_map_info = None
        self.anomaly_pose = None
        self.laser_data = np.zeros(360) + 3.5
        self.last_scan_msg = None
        self.last_action_nav = np.zeros(2) 
        self.last_action_patrol = np.zeros(3)
        self.task_running = False
        self.anomaly_spawned = False
        self.num_scan_points = 24
        self.anomaly_cost_threshold = 50 
        self.consecutive_points_threshold = 7
        self.anomaly_model_name = "anomaly_box"
        self.goal_model_name = "visual_goal"

        self.is_simulation = rospy.get_param('/use_sim_time', False)
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('blender_map_project')
        
        areas_yaml_path = os.path.join(pkg_path, 'maps', 'patrol_areas.yaml') if self.is_simulation else os.path.join(pkg_path, 'maps', 'patrol_areas_real.yaml')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"--- 使用的計算設備: {self.device} ---")
        
        nav_models_dir = os.path.join(pkg_path, 'models_checkpoints')
        self.sac_nav_policy = self._load_sac_model(nav_models_dir, "導航模型 (第一階段)", policy_class=PolicyNetworkStage1, state_dim=31, action_dim=2)

        patrol_models_dir = os.path.join(pkg_path, 'models_checkpoints_stage3')
        self.sac_patrol_policy = self._load_sac_model(patrol_models_dir, "巡邏模型 (第三階段)", policy_class=PolicyNetworkStage3, state_dim=32, action_dim=3)

        if self.sac_nav_policy is None or self.sac_patrol_policy is None:
            rospy.signal_shutdown("模型載入失敗，程式終止。")
            return

        try:
            with open(areas_yaml_path, 'r') as f:
                self.patrol_areas = yaml.safe_load(f)
                rospy.loginfo(f"成功載入巡邏區域設定檔: {os.path.basename(areas_yaml_path)}")
        except FileNotFoundError:
            rospy.logerr(f"錯誤：找不到巡邏區域設定檔！路徑: {areas_yaml_path}"); rospy.signal_shutdown("設定檔載入失敗"); return

        # 建立發佈者與訂閱者
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.pub_markers = rospy.Publisher("/patrol_visuals", MarkerArray, queue_size=1)
        # 異常分數發佈者
        self.pub_anomaly_score = rospy.Publisher("/patrol/anomaly_score", Float32, queue_size=10)
        
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self._odom_callback)
        self.sub_scan = rospy.Subscriber("/scan", LaserScan, self._scan_callback)
        self.static_map_sub = rospy.Subscriber("/map", OccupancyGrid, self._static_map_callback)
        self.tf_listener = tf.TransformListener()

        if self.is_simulation:
            rospy.loginfo("正在啟用Gazebo服務...")
            rospy.wait_for_service('/gazebo/set_model_state'); self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.wait_for_service('/gazebo/spawn_sdf_model'); self.spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            rospy.wait_for_service('/gazebo/delete_model'); self.delete_model_proxy = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            try:
                with open(os.path.join(pkg_path, 'models', 'anomaly_box', 'model.sdf'), 'r') as f: self.anomaly_model_sdf = f.read()
            except Exception as e: rospy.logwarn(f"無法載入 anomaly_box 模型: {e}"); self.anomaly_model_sdf = None
        
        rospy.loginfo("--- 指揮中心初始化完成，待命中 ---")

    def _load_sac_model(self, model_dir, model_name, policy_class, state_dim, action_dim):
        rospy.loginfo(f"正在載入 {model_name}...")
        checkpoints = glob.glob(os.path.join(model_dir, "checkpoint_ep*.pth"))
        if not checkpoints:
            rospy.logerr(f"錯誤：在目錄 {model_dir} 中找不到任何模型檔案！")
            return None
        
        latest_episode = max([int(f.split('ep')[1].split('.pth')[0]) for f in checkpoints])
        model_path = os.path.join(model_dir, f'checkpoint_ep{latest_episode}.pth')
        rospy.loginfo(f"  已自動選擇最新模型：Ep {latest_episode}")

        policy = policy_class(state_dim, action_dim).to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            policy.eval()
            rospy.loginfo(f"  成功載入模型: {model_path}")
            return policy
        except Exception as e:
            rospy.logerr(f"  錯誤：載入SAC模型檔案失敗！路徑: {model_path}, 錯誤訊息: {e}")
            return None
            
    def _odom_callback(self, msg):
        self.robot_velocity = msg.twist.twist

    def _scan_callback(self, msg):
        ranges = np.array(msg.ranges); ranges[np.isinf(ranges)] = 3.5; ranges[np.isnan(ranges)] = 3.5
        self.laser_data = ranges; self.last_scan_msg = msg

    def _static_map_callback(self, msg):
        if self.static_map_data is None:
            self.static_map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
            self.static_map_info = msg.info
            self.static_map_sub.unregister() 
            rospy.loginfo("已成功接收並儲存靜態地圖。")

    def _update_robot_pose_from_tf(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            self.robot_pose.position.x, self.robot_pose.position.y = trans[0], trans[1]
            self.robot_pose.orientation.x, self.robot_pose.orientation.y, self.robot_pose.orientation.z, self.robot_pose.orientation.w = rot[0], rot[1], rot[2], rot[3]
            return True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn_throttle(5.0, "獲取 TF 轉換失敗 (map -> base_footprint)。")
            return False

    def goto_area_center(self, area_info):
        target_point = (area_info['center_x'], area_info['center_y'])
        rospy.loginfo(f"命令已下達：使用導航模型前往目標區域中心 ({target_point[0]:.2f}, {target_point[1]:.2f})")
        self._publish_all_visuals(nav_target=target_point)
        self._update_gazebo_goal_marker(target_point)
        timeout = time.time() + 120
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown() and time.time() < timeout:
            self.pub_anomaly_score.publish(0.0) # 導航時分數為0
            if not self._update_robot_pose_from_tf():
                rate.sleep(); continue

            if self._calculate_dist_to_goal(target_point) < 0.3:
                rospy.loginfo("導航模型回報：已成功到達區域中心。")
                self.pub_cmd_vel.publish(Twist()); return True
                
            state = self._get_nav_state(target_point)
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = self.sac_nav_policy.forward(state_tensor)
                action = torch.tanh(mean).detach().cpu().numpy()[0]

            vel_cmd = Twist(); vel_cmd.linear.x = (action[0] + 1) / 2 * 0.22; vel_cmd.angular.z = action[1] * 1.8
            self.pub_cmd_vel.publish(vel_cmd)
            self.last_action_nav = action
            rate.sleep()
            
        rospy.logerr("導航模型回報：前往區域中心失敗 (超時)！")
        self.pub_cmd_vel.publish(Twist()); return False

    def run_sac_patrol(self, area_id, area_info, spawn_anomaly):
        rospy.loginfo(f"大腦已切換至巡邏模型，開始在 {area_id} 執行精細巡邏...")
        w, h = area_info['width'] / 2, area_info['height'] / 2; cx, cy = area_info['center_x'], area_info['center_y']
        waypoints = [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]
        
        if spawn_anomaly:
            if self.is_simulation and self.anomaly_model_sdf: self._spawn_anomaly_in_sim(area_info, waypoints)
            else: rospy.loginfo("="*50 + "\n實機測試提示：請在接下來的 10 秒內放置障礙物。\n" + "="*50); time.sleep(10)
        else: rospy.loginfo("本回合不生成異常。")

        for i, wp in enumerate(waypoints):
            rospy.loginfo(f"巡邏模型正在前往航點 {i+1}/{len(waypoints)}: ({wp[0]:.2f}, {wp[1]:.2f})")
            self._publish_all_visuals(area_id, area_info, waypoints, current_waypoint_index=i)
            self._update_gazebo_goal_marker(wp)
            timeout = time.time() + 60
            rate = rospy.Rate(10)
            
            while not rospy.is_shutdown() and time.time() < timeout:
                if not self._update_robot_pose_from_tf():
                    rate.sleep(); continue

                if self._calculate_dist_to_goal(wp) < 0.3:
                    rospy.loginfo(f"成功到達航點 {i+1}。"); break
                    
                state = self._get_patrol_state(wp)
                state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                
                with torch.no_grad():
                    mean, _, anomaly_score_tensor = self.sac_patrol_policy.forward(state_tensor)
                    driving_action = torch.tanh(mean).detach().cpu().numpy()[0]
                    anomaly_score = anomaly_score_tensor.detach().cpu().numpy()[0][0]
                
                # (+) 發佈最新的異常分數
                self.pub_anomaly_score.publish(anomaly_score)

                vel_cmd = Twist(); vel_cmd.linear.x = (driving_action[0] + 1) / 2 * 0.22; vel_cmd.angular.z = driving_action[1] * 1.8
                self.pub_cmd_vel.publish(vel_cmd)
                
                if anomaly_score > 0.8:
                    rospy.logwarn(f"偵測到潛在異常！(分數: {anomaly_score:.2f})")

                self.last_action_patrol = np.array([driving_action[0], driving_action[1], anomaly_score])
                rate.sleep()
            else:
                rospy.logwarn(f"前往航點 {i+1} 超時！"); self.pub_cmd_vel.publish(Twist()); return False
        
        rospy.loginfo(f"區域 {area_id} 所有航點巡邏完畢！"); self.pub_cmd_vel.publish(Twist()); return True
    
    def run_mission(self, area_id, area_info):
        self.task_running = True
        rospy.loginfo(f"--- 新任務開始: {area_id} ---")
        self._cleanup_previous_task()
        
        if self.goto_area_center(area_info):
            spawn_anomaly = False
            try:
                # 使用簡單的 input，因為它在獨立執行緒中是安全的
                choice = input(f"已到達區域 {area_id} 中心。是否要生成異常? (y/n): ").lower().strip()
                if choice == 'y':
                    spawn_anomaly = True
            except Exception:
                pass # 忽略 ctrl+c 等錯誤
            
            self.run_sac_patrol(area_id, area_info, spawn_anomaly)
        
        rospy.loginfo(f"--- 任務 {area_id} 流程結束，返回待命狀態 ---")
        self._cleanup_previous_task()
        self.pub_anomaly_score.publish(0.0) # 任務結束後重置分數
        self.task_running = False

    def _input_thread_func(self):
        """專門處理使用者輸入的執行緒函式"""
        while not rospy.is_shutdown():
            if self.task_running:
                time.sleep(1); continue

            self._show_menu()
            try:
                choice = input("請輸入指令: ").lower().strip()
                if not choice: continue
                if choice == 'q':
                    rospy.signal_shutdown("使用者要求退出。"); break
                
                area_id, area_info = None, None
                if choice == 's':
                    area_id = random.choice(list(self.patrol_areas.keys()))
                else:
                    area_index = int(choice) - 1
                    area_keys = list(self.patrol_areas.keys())
                    if 0 <= area_index < len(area_keys):
                        area_id = area_keys[area_index]
                
                if area_id:
                    area_info = self.patrol_areas[area_id]
                    # 在新執行緒中執行任務，避免阻塞輸入
                    mission_thread = threading.Thread(target=self.run_mission, args=(area_id, area_info))
                    mission_thread.daemon = True
                    mission_thread.start()
                else:
                    print("\n無效的數字編號。")

            except (ValueError, IndexError):
                print("\n輸入錯誤，請輸入列表中的數字編號或 's'/'q'。")
            except Exception:
                rospy.loginfo("\n輸入執行緒已關閉。")
                break

    def main_loop(self):
        """主迴圈，啟動輸入執行緒並維持ROS節點運行。"""
        input_thread = threading.Thread(target=self._input_thread_func)
        input_thread.daemon = True
        input_thread.start()
        rospy.spin()
        rospy.loginfo("--- 指揮中心已關閉 ---")
    
    def _show_menu(self):
        print("\n" + "="*40 + "\n--- 巡邏任務指揮中心 ---\n" + "="*40)
        for i, key in enumerate(self.patrol_areas.keys()): 
            print(f"[{i+1}] 巡邏區域: {key}")
        print("[s] 隨機區域巡邏")
        print("[q] 退出程式")
    
    def _cleanup_previous_task(self):
        if self.is_simulation and self.anomaly_spawned:
            self._delete_anomaly_in_sim()
        self.pub_markers.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        self.pub_cmd_vel.publish(Twist())

    def _update_gazebo_goal_marker(self, target_point):
        if not self.is_simulation: return
        try:
            goal_pose = Pose()
            goal_pose.position.x = target_point[0]
            goal_pose.position.y = target_point[1]
            goal_pose.position.z = 0.3
            goal_state = ModelState(model_name=self.goal_model_name, pose=goal_pose)
            self.set_model_state_proxy(goal_state)
        except rospy.ServiceException as e: 
            rospy.logwarn_throttle(10, f"更新 Gazebo 目標點標記失敗: {e}")

    def _get_nav_state(self, target_point):
        current_scan = self._get_downsampled_laser(self.laser_data)
        angle_to_goal = self._calculate_angle_to_goal(target_point)
        dist_to_goal = self._calculate_dist_to_goal(target_point)
        return np.concatenate([
            current_scan / 3.5, np.array([dist_to_goal / 10.0, angle_to_goal / math.pi, 0.0]), 
            self.last_action_nav, np.array([self.robot_velocity.linear.x, self.robot_velocity.angular.z])
        ])

    def _get_patrol_state(self, current_waypoint):
        current_scan = self._get_downsampled_laser(self.laser_data)
        angle_to_goal = self._calculate_angle_to_goal(current_waypoint)
        dist_to_goal = self._calculate_dist_to_goal(current_waypoint)
        return np.concatenate([
            current_scan / 3.5,
            np.array([dist_to_goal / 10.0, angle_to_goal / math.pi, 0.0]), 
            self.last_action_patrol, 
            np.array([self.robot_velocity.linear.x, self.robot_velocity.angular.z])
        ])
        
    def _calculate_angle_to_goal(self, waypoint):
        goal_angle = math.atan2(waypoint[1] - self.robot_pose.position.y, waypoint[0] - self.robot_pose.position.x)
        _, _, yaw = euler_from_quaternion([self.robot_pose.orientation.x, self.robot_pose.orientation.y, self.robot_pose.orientation.z, self.robot_pose.orientation.w])
        return self.normalize_angle(goal_angle - yaw)

    def _calculate_dist_to_goal(self, waypoint):
        return math.hypot(waypoint[0] - self.robot_pose.position.x, waypoint[1] - self.robot_pose.position.y)

    def _spawn_anomaly_in_sim(self, area_info, waypoints):
        spawn_attempts = 0
        while spawn_attempts < 20:
            pose = Pose(); pose.position.x = random.uniform(area_info['center_x']-area_info['width']/2, area_info['center_x']+area_info['width']/2); pose.position.y = random.uniform(area_info['center_y']-area_info['height']/2, area_info['center_y']+area_info['height']/2)
            is_safe = all(math.hypot(pose.position.x - wp[0], pose.position.y - wp[1]) > 0.5 for wp in waypoints) and \
                      math.hypot(pose.position.x - self.robot_pose.position.x, pose.position.y - self.robot_pose.position.y) > 0.6
            if is_safe:
                try:
                    self.spawn_model_proxy(self.anomaly_model_name, self.anomaly_model_sdf, "", pose, "world")
                    self.anomaly_spawned = True
                    self.anomaly_pose = pose
                    rospy.loginfo(f"[模擬] 已在安全位置生成異常障礙物！")
                    self._publish_all_visuals()
                    return
                except rospy.ServiceException as e: rospy.logerr(f"生成異常障礙物失敗: {e}"); return
            spawn_attempts += 1
        rospy.logwarn("[模擬] 嘗試多次後未找到安全生成點，本回合無異常。")

    def _delete_anomaly_in_sim(self):
        try:
            self.delete_model_proxy(self.anomaly_model_name)
            self.anomaly_spawned = False
            self.anomaly_pose = None
            rospy.loginfo("[模擬] 已清除異常障礙物。")
            self._publish_all_visuals()
        except rospy.ServiceException: pass

    def _get_downsampled_laser(self, full_scan):
        step = len(full_scan) // self.num_scan_points
        return np.array([np.percentile(full_scan[i*step:(i+1)*step], 20) for i in range(self.num_scan_points)])

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def _publish_all_visuals(self, area_id=None, area_info=None, waypoints=None, current_waypoint_index=-1, nav_target=None):
        marker_array = MarkerArray()
        
        if area_id and area_info and waypoints:
            box_marker = Marker(); box_marker.header.frame_id = "map"; box_marker.header.stamp = rospy.Time.now(); box_marker.ns = "patrol_area"; box_marker.id = 0; box_marker.type = Marker.LINE_STRIP; box_marker.action = Marker.ADD
            box_marker.scale.x = 0.05; box_marker.color.g = 1.0; box_marker.color.a = 1.0
            cx, cy, w, h = area_info['center_x'], area_info['center_y'], area_info['width'] / 2, area_info['height'] / 2
            box_points = [(cx-w, cy-h), (cx+w, cy-h), (cx+w, cy+h), (cx-w, cy+h), (cx-w, cy-h)]
            for p in box_points: point_msg = Point(); point_msg.x, point_msg.y = p[0], p[1]; box_marker.points.append(point_msg)
            marker_array.markers.append(box_marker)
            
            for i, wp in enumerate(waypoints):
                wp_marker = Marker(); wp_marker.header.frame_id = "map"; wp_marker.header.stamp = rospy.Time.now(); wp_marker.ns = "waypoints"; wp_marker.id = i + 1; wp_marker.type = Marker.SPHERE; wp_marker.action = Marker.ADD
                wp_marker.pose.position.x, wp_marker.pose.position.y, wp_marker.pose.position.z = wp[0], wp[1], 0.1
                if i == current_waypoint_index:
                    wp_marker.scale.x, wp_marker.scale.y, wp_marker.scale.z = 0.25, 0.25, 0.25; wp_marker.color.r = 1.0; wp_marker.color.g = 1.0; wp_marker.color.b = 1.0; wp_marker.color.a = 1.0
                else:
                    wp_marker.scale.x, wp_marker.scale.y, wp_marker.scale.z = 0.2, 0.2, 0.2; wp_marker.color.r = 0.5; wp_marker.color.g = 0.8; wp_marker.color.b = 1.0; wp_marker.color.a = 1.0
                marker_array.markers.append(wp_marker)
                
                label_marker = Marker(); label_marker.header.frame_id = "map"; label_marker.header.stamp = rospy.Time.now(); label_marker.ns = "waypoint_labels"; label_marker.id = i + 10; label_marker.type = Marker.TEXT_VIEW_FACING; label_marker.action = Marker.ADD
                label_marker.pose.position.x, label_marker.pose.position.y, label_marker.pose.position.z = wp[0], wp[1], 0.4; label_marker.scale.z = 0.3; label_marker.color.r, label_marker.color.g, label_marker.color.b, label_marker.color.a = 1.0, 1.0, 1.0, 1.0
                label_marker.text = f"P{i+1}\n({wp[0]:.2f}, {wp[1]:.2f})"
                marker_array.markers.append(label_marker)
            
            area_label_marker = Marker(); area_label_marker.header.frame_id = "map"; area_label_marker.header.stamp = rospy.Time.now(); area_label_marker.ns = "area_label"; area_label_marker.id = 100; area_label_marker.type = Marker.TEXT_VIEW_FACING; area_label_marker.action = Marker.ADD
            area_label_marker.pose.position.x, area_label_marker.pose.position.y, area_label_marker.pose.position.z = cx, cy, 1.0; area_label_marker.scale.z = 0.5; area_label_marker.color.r, area_label_marker.color.g, area_label_marker.color.b, area_label_marker.color.a = 1.0, 1.0, 0.0, 1.0
            area_label_marker.text = f"Area: {area_id}"
            marker_array.markers.append(area_label_marker)

        if nav_target:
            marker = Marker(); marker.header.frame_id = "map"; marker.header.stamp = rospy.Time.now(); marker.ns = "navigation_goal"; marker.id = 200; marker.type = Marker.ARROW; marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = nav_target[0], nav_target[1], 1.0
            q = quaternion_from_euler(-math.pi/2, 0, 0)
            marker.pose.orientation = Quaternion(*q)
            marker.scale.x = 1.0; marker.scale.y = 0.2; marker.scale.z = 0.2
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 1.0
            marker.lifetime = rospy.Duration(0)
            marker_array.markers.append(marker)

        if self.anomaly_spawned and self.anomaly_pose is not None:
            anomaly_marker = Marker(); anomaly_marker.header.frame_id = "map"; anomaly_marker.header.stamp = rospy.Time.now(); anomaly_marker.ns = "anomaly_object"; anomaly_marker.id = 300
            anomaly_marker.type = Marker.CUBE; anomaly_marker.action = Marker.ADD; anomaly_marker.pose = self.anomaly_pose
            anomaly_marker.scale.x, anomaly_marker.scale.y, anomaly_marker.scale.z = 0.3, 0.3, 0.3
            anomaly_marker.color.r, anomaly_marker.color.g, anomaly_marker.color.b, anomaly_marker.color.a = 1.0, 0.2, 0.2, 0.8
            marker_array.markers.append(anomaly_marker)

        self.pub_markers.publish(marker_array)

if __name__ == '__main__':
    try:
        manager = PatrolManager()
        manager.main_loop()
    except rospy.ROSInterruptException:
        pass

