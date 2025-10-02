#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
import os
import numpy as np
import math
import time
import glob
import random
import threading

# ROS 訊息與服務
from geometry_msgs.msg import Twist, Pose, Quaternion, Point, PoseStamped  # MODIFIED: 引入 PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from gazebo_msgs.srv import SetModelState, SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState

# PyTorch (用於載入和執行SAC模型)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 從訓練腳本中引入兩個不同的網路架構
from train_sac_stage3 import PolicyNetworkStage3, weights_init_

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
        rospy.init_node('patrol_manager')
        rospy.loginfo("--- 巡邏任務指揮中心 初始化 (RViz 動態目標模式) ---")

        self.is_simulation = rospy.get_param('/use_sim_time', False)
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('blender_map_project')
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"--- 使用的計算設備: {self.device} ---")
        
        nav_models_dir = os.path.join(pkg_path, 'models_checkpoints')
        self.sac_nav_policy = self._load_sac_model(nav_models_dir, "導航模型 (第一階段)", policy_class=PolicyNetworkStage1, state_dim=31, action_dim=2)

        patrol_models_dir = os.path.join(pkg_path, 'models_checkpoints_stage3')
        self.sac_patrol_policy = self._load_sac_model(patrol_models_dir, "巡邏模型 (第三階段)", policy_class=PolicyNetworkStage3, state_dim=56, action_dim=3)

        if self.sac_nav_policy is None or self.sac_patrol_policy is None:
            rospy.signal_shutdown("模型載入失敗，程式終止。")
            return

        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.pub_markers = rospy.Publisher("/patrol_visuals", MarkerArray, queue_size=1)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self._odom_callback)
        self.sub_scan = rospy.Subscriber("/scan", LaserScan, self._scan_callback)
        self.sub_rviz_goal = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._rviz_goal_callback)

        self.anomaly_model_name = "anomaly_box"; self.goal_model_name = "visual_goal"; self.anomaly_spawned = False
        if self.is_simulation:
            rospy.loginfo("正在啟用Gazebo服務...")
            rospy.wait_for_service('/gazebo/set_model_state'); self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.wait_for_service('/gazebo/spawn_sdf_model'); self.spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            rospy.wait_for_service('/gazebo/delete_model'); self.delete_model_proxy = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            try:
                with open(os.path.join(pkg_path, 'models', 'anomaly_box', 'model.sdf'), 'r') as f: self.anomaly_model_sdf = f.read()
            except Exception as e: rospy.logwarn(f"無法載入 anomaly_box 模型: {e}"); self.anomaly_model_sdf = None

        self.robot_pose = Pose(); self.robot_velocity = Twist()
        self.laser_data = np.zeros(360) + 3.5
        self.num_scan_points = 24
        self.baseline_scan = np.zeros(self.num_scan_points)
        self.last_action_nav = np.zeros(2) 
        self.last_action_patrol = np.zeros(3)
        self.task_running = False

        rospy.loginfo("--- 指揮中心初始化完成，待命中 ---")

    # --- MODIFIED: 新增的核心回呼函式 ---
    def _rviz_goal_callback(self, msg):
        """
        當在 RViz 中使用 "2D Nav Goal" 時，此函式會被觸發。
        """
        if self.task_running:
            rospy.logwarn("目前有任務正在執行中，請等待任務結束後再指定新目標。")
            return

        rospy.loginfo("="*40)
        rospy.loginfo("收到來自 RViz 的新巡邏任務指令！")
        
        # 從訊息中提取中心點座標
        center_x = msg.pose.position.x
        center_y = msg.pose.position.y

        # 動態生成 area_info 字典
        area_info = {
            'center_x': center_x,
            'center_y': center_y,
            'width': 1.5,
            'height': 1.5
        }
        
        # 建立一個獨特的任務ID
        area_id = f"rviz_task_{int(time.time())}"

        # 使用執行緒來啟動任務，避免阻塞回呼函式
        mission_thread = threading.Thread(target=self.run_mission, args=(area_id, area_info))
        mission_thread.daemon = True
        mission_thread.start()

    def main_loop(self):
        """
        主迴圈現在只做等待，真正的任務啟動點在 _rviz_goal_callback
        """
        rospy.loginfo("\n" + "="*40 + "\n--- 巡邏任務指揮中心 ---\n" + "="*40)
        rospy.loginfo("系統待命中...")
        rospy.loginfo("請在 RViz 中使用 [2D Nav Goal] 工具來指定巡邏區域的中心點。")
        
        # rospy.spin() 會讓節點保持運行，直到被關閉
        rospy.spin()
        
        rospy.loginfo("--- 指揮中心已關閉 ---")


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
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            policy.eval()
            rospy.loginfo(f"  成功載入模型: {model_path}")
            return policy
        except Exception as e:
            rospy.logerr(f"  錯誤：載入SAC模型檔案失敗！路徑: {model_path}, 錯誤訊息: {e}")
            return None

    def _odom_callback(self, msg):
        self.robot_pose = msg.pose.pose; self.robot_velocity = msg.twist.twist

    def _scan_callback(self, msg):
        ranges = np.array(msg.ranges); ranges[np.isinf(ranges)] = 3.5; ranges[np.isnan(ranges)] = 3.5
        self.laser_data = ranges

    def goto_area_center(self, area_info):
        target_point = (area_info['center_x'], area_info['center_y'])
        rospy.loginfo(f"命令已下達：使用導航模型前往目標區域中心 ({target_point[0]:.2f}, {target_point[1]:.2f})")
        self._publish_nav_goal_marker(target_point); self._update_gazebo_goal_marker(target_point)
        timeout = time.time() + 120
        while not rospy.is_shutdown() and time.time() < timeout:
            if self._calculate_dist_to_goal(target_point) < 0.3:
                rospy.loginfo("導航模型回報：已成功到達區域中心。")
                self.pub_cmd_vel.publish(Twist()); return True
            state = self._get_nav_state(target_point)
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _ = self.sac_nav_policy.sample(state_tensor)
            action = action_tensor.detach().cpu().numpy()[0]
            vel_cmd = Twist(); vel_cmd.linear.x = (action[0] + 1) / 2 * 0.22; vel_cmd.angular.z = action[1] * 2.0
            self.pub_cmd_vel.publish(vel_cmd)
            self.last_action_nav = action
            rospy.sleep(0.1)
        rospy.logerr("導航模型回報：前往區域中心失敗 (超時)！")
        self.pub_cmd_vel.publish(Twist()); return False

    def run_sac_patrol(self, area_id, area_info, spawn_anomaly):
        rospy.loginfo(f"大腦已切換至巡邏模型，開始在 {area_id} 執行精細巡邏...")
        w, h = area_info['width'] / 2, area_info['height'] / 2; cx, cy = area_info['center_x'], area_info['center_y']
        waypoints = [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]
        
        rospy.loginfo("正在執行基準掃描..."); time.sleep(1)
        self.baseline_scan = self._get_downsampled_laser(self.laser_data); rospy.loginfo("基準掃描完成。")

        if spawn_anomaly:
            if self.is_simulation:
                if self.anomaly_model_sdf: self._spawn_anomaly_in_sim(area_info, waypoints)
                else: rospy.loginfo("[模擬] 障礙物模型未載入，無法生成。")
            else:
                rospy.loginfo("="*50 + "\n實機測試提示：請在接下來的 10 秒內放置障礙物。\n" + "="*50)
                time.sleep(10); rospy.loginfo("時間到，開始巡邏。")
        else:
            rospy.loginfo("本回合不生成異常。")

        for i, wp in enumerate(waypoints):
            rospy.loginfo(f"巡邏模型正在前往航點 {i+1}/{len(waypoints)}: ({wp[0]:.2f}, {wp[1]:.2f})")
            self._publish_patrol_markers(area_id, area_info, waypoints, current_waypoint_index=i)
            self._update_gazebo_goal_marker(wp)
            timeout = time.time() + 60
            while not rospy.is_shutdown() and time.time() < timeout:
                if self._calculate_dist_to_goal(wp) < 0.3:
                    rospy.loginfo(f"成功到達航點 {i+1}。"); break
                state = self._get_patrol_state(wp)
                state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, _ = self.sac_patrol_policy.sample(state_tensor)
                action = action_tensor.detach().cpu().numpy()[0]
                vel_cmd = Twist(); vel_cmd.linear.x = (action[0] + 1) / 2 * 0.22; vel_cmd.angular.z = action[1] * 2.0
                self.pub_cmd_vel.publish(vel_cmd)
                if action[2] > 0.8: rospy.logwarn(f"巡邏模型偵測到潛在異常！分數: {action[2]:.2f}")
                self.last_action_patrol = action
                rospy.sleep(0.1)
            else:
                rospy.logwarn(f"前往航點 {i+1} 超時！"); self.pub_cmd_vel.publish(Twist()); return False
        
        rospy.loginfo(f"區域 {area_id} 所有航點巡邏完畢！"); self.pub_cmd_vel.publish(Twist()); return True

    def run_mission(self, area_id, area_info):
        self.task_running = True
        rospy.loginfo("--- 新任務開始 ---")
        self._cleanup_previous_task()
        
        if self.goto_area_center(area_info):
            spawn_anomaly = False
            # 在非模擬環境中，總是詢問是否生成異常
            if not self.is_simulation:
                while not rospy.is_shutdown():
                    try:
                        anomaly_choice = input("已到達區域中心。是否要模擬異常(手動放置障礙物)? (y/n): ").lower().strip()
                        if anomaly_choice in ['y', 'n']:
                            spawn_anomaly = (anomaly_choice == 'y')
                            break
                        else:
                            print("輸入無效，請輸入 'y' 或 'n'。")
                    except (EOFError, KeyboardInterrupt): 
                        rospy.logwarn("未收到輸入或用戶中斷，默認不生成異常。")
                        break
            # 在模擬環境中，也可以選擇詢問
            elif self.is_simulation:
                spawn_anomaly = True # 或者您可以保留 input 詢問

            self.run_sac_patrol(area_id, area_info, spawn_anomaly=spawn_anomaly)
        
        rospy.loginfo("--- 任務流程結束，返回待命狀態 ---")
        self._cleanup_previous_task()
        self.task_running = False
        rospy.sleep(1)
        rospy.loginfo("\n" + "="*40 + "\n系統待命中... 請在 RViz 中指定下一個巡邏中心點。\n" + "="*40)
    
    # --- MODIFIED: _show_menu 函式已被移除 ---
    
    def _cleanup_previous_task(self):
        if self.is_simulation and self.anomaly_spawned:
            self._delete_anomaly_in_sim()
        self.pub_markers.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        self.pub_cmd_vel.publish(Twist())

    def _update_gazebo_goal_marker(self, target_point):
        if not self.is_simulation: return
        try:
            goal_pose = Pose(); goal_pose.position.x = target_point[0]; goal_pose.position.y = target_point[1]; goal_pose.position.z = 0.3
            goal_state = ModelState(model_name=self.goal_model_name, pose=goal_pose)
            self.set_model_state_proxy(goal_state)
        except rospy.ServiceException as e: rospy.logwarn(f"更新 Gazebo 目標點標記失敗: {e}")

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
        diff_scan = current_scan - self.baseline_scan
        angle_to_goal = self._calculate_angle_to_goal(current_waypoint)
        dist_to_goal = self._calculate_dist_to_goal(current_waypoint)
        return np.concatenate([
            current_scan / 3.5,
            diff_scan, 
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
                    self.spawn_model_proxy(self.anomaly_model_name, self.anomaly_model_sdf, "", pose, "world"); self.anomaly_spawned = True; rospy.loginfo(f"[模擬] 已在安全位置生成異常障礙物！")
                    return
                except rospy.ServiceException as e: rospy.logerr(f"生成異常障礙物失敗: {e}"); return
            spawn_attempts += 1
        rospy.logwarn("[模擬] 嘗試多次後未找到安全生成點，本回合無異常。")

    def _delete_anomaly_in_sim(self):
        try:
            self.delete_model_proxy(self.anomaly_model_name); self.anomaly_spawned = False; rospy.loginfo("[模擬] 已清除異常障礙物。")
        except rospy.ServiceException: pass

    def _get_downsampled_laser(self, full_scan):
        step = len(full_scan) // self.num_scan_points
        return np.array([min(full_scan[i*step:(i+1)*step]) for i in range(self.num_scan_points)])

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def _publish_nav_goal_marker(self, target_point):
        marker = Marker(); marker.header.frame_id = "map"; marker.header.stamp = rospy.Time.now(); marker.ns = "navigation_goal"; marker.id = 0; marker.type = Marker.ARROW; marker.action = Marker.ADD
        marker.pose.position.x = target_point[0]; marker.pose.position.y = target_point[1]; marker.pose.position.z = 1.0
        # MODIFIED: 箭頭方向修正
        q = quaternion_from_euler(-math.pi/2, 0, 0) 
        marker.pose.orientation = Quaternion(*q)
        marker.scale.x = 1.0; marker.scale.y = 0.2; marker.scale.z = 0.2
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 1.0
        marker.lifetime = rospy.Duration(0)
        self.pub_markers.publish(MarkerArray(markers=[marker]))

    def _publish_patrol_markers(self, area_id, area_info, waypoints, current_waypoint_index=-1):
        marker_array = MarkerArray()
        
        box_marker = Marker()
        box_marker.header.frame_id = "map"
        box_marker.header.stamp = rospy.Time.now()
        box_marker.ns = "patrol_area"
        box_marker.id = 0
        box_marker.type = Marker.CUBE # <-- 改為 CUBE
        box_marker.action = Marker.ADD
        
        # CUBE 的位置是中心點
        box_marker.pose.position.x = area_info['center_x']
        box_marker.pose.position.y = area_info['center_y']
        box_marker.pose.position.z = 0
        box_marker.pose.orientation.w = 1.0
        
        # CUBE 的尺寸是長寬高
        box_marker.scale.x = area_info['width']
        box_marker.scale.y = area_info['height']
        box_marker.scale.z = 0.01

        # 顏色設為半透明綠色
        box_marker.color.g = 1.0
        box_marker.color.a = 0.3 # 半透明
        
        marker_array.markers.append(box_marker)
        
        for i, wp in enumerate(waypoints):
            wp_marker = Marker(); wp_marker.header.frame_id = "map"; wp_marker.header.stamp = rospy.Time.now(); wp_marker.ns = "waypoints"; wp_marker.id = i + 1; wp_marker.type = Marker.SPHERE; wp_marker.action = Marker.ADD
            wp_marker.pose.position.x, wp_marker.pose.position.y, wp_marker.pose.position.z = wp[0], wp[1], 0.1
            
            if i == current_waypoint_index:
                wp_marker.scale.x, wp_marker.scale.y, wp_marker.scale.z = 0.25, 0.25, 0.25
                wp_marker.color.r, wp_marker.color.g, wp_marker.color.b, wp_marker.color.a = 1.0, 1.0, 1.0, 1.0
            else:
                wp_marker.scale.x, wp_marker.scale.y, wp_marker.scale.z = 0.2, 0.2, 0.2
                wp_marker.color.r, wp_marker.color.g, wp_marker.color.b, wp_marker.color.a = 0.5, 0.8, 1.0, 1.0
            marker_array.markers.append(wp_marker)
            
            label_marker = Marker(); label_marker.header.frame_id = "map"; label_marker.header.stamp = rospy.Time.now(); label_marker.ns = "waypoint_labels"; label_marker.id = i + 10; label_marker.type = Marker.TEXT_VIEW_FACING; label_marker.action = Marker.ADD
            label_marker.pose.position.x, label_marker.pose.position.y, label_marker.pose.position.z = wp[0], wp[1], 0.4; label_marker.scale.z = 0.3; label_marker.color.r, label_marker.color.g, label_marker.color.b, label_marker.color.a = 1.0, 1.0, 1.0, 1.0
            label_marker.text = f"P{i+1}"
            marker_array.markers.append(label_marker)
        
        area_label_marker = Marker(); area_label_marker.header.frame_id = "map"; area_label_marker.header.stamp = rospy.Time.now(); area_label_marker.ns = "area_label"; area_label_marker.id = 100; area_label_marker.type = Marker.TEXT_VIEW_FACING; area_label_marker.action = Marker.ADD
        area_label_marker.pose.position.x, area_label_marker.pose.position.y, area_label_marker.pose.position.z = area_info['center_x'], area_info['center_y'], 1.0; area_label_marker.scale.z = 0.5; area_label_marker.color.r, area_label_marker.color.g, area_label_marker.color.b, area_label_marker.color.a = 1.0, 1.0, 0.0, 1.0
        area_label_marker.text = f"Area: {area_id}"
        marker_array.markers.append(area_label_marker)

        self.pub_markers.publish(marker_array)

if __name__ == '__main__':
    try:
        manager = PatrolManager()
        manager.main_loop()
    except rospy.ROSInterruptException:
        pass
