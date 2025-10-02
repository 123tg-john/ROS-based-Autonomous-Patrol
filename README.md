# 基於深度強化學習之自主巡邏與異常偵測機器人 (Autonomous Patrol and Anomaly Detection Robot using DRL)

[![ROS Noetic](https://img.shields.io/badge/ROS-Noetic-blue)](http://wiki.ros.org/noetic)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange)](https://pytorch.org/)

本專案旨在開發一套能夠在複雜環境中，同時執行自主巡邏及即時異常偵測的智慧機器人系統。系統基於 **ROS (Robot Operating System)** 框架，並採用 **深度強化學習 (Deep Reinforcement Learning)** 中的 **Soft Actor-Critic (SAC)** 演算法進行模型訓練。

---

## 系統展示 (Demonstration)

[![專案展示影片](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID)
*<p align="center">點擊圖片觀看實機操作展示影片</p>*


## 核心功能 (Key Features)

* **整合式系統架構**: 以 ROS 為核心，整合 Gazebo 模擬、實體機器人與 Web 前端，實現從訓練、模擬到部署的完整流程。
* **多任務學習 Agent**: 基於 SAC 演算法，訓練單一 Agent 同時完成「自主導航」與「異常偵測」兩項任務。
* **優化的訓練策略**: 應用 **遷移學習 (Transfer Learning)** 與 **課程學習 (Curriculum Learning)**，顯著提升訓練效率與模型的最終性能。
* **兩種偵測方法論**: 實現並深入比較了兩種異常感知機制：
    1.  **代價地圖法 (Costmap-based)**：利用環境先驗知識，穩定性高。
    2.  **基線比對法 (Baseline Comparison)**：不依賴地圖，泛化潛力強。
* **人性化人機互動介面**: 開發了基於 Web 的遠端監控介面，可即時觀看機器人狀態、地圖、攝影機畫面，並下達巡邏指令。

## 系統架構 (System Architecture)

本系統由前端使用者介面、ROS 後端系統，以及物理/模擬環境三大部分組成，透過 WebSocket 進行通訊。

![系統架構圖](<img width="773" height="929" alt="模擬流程圖" src="https://github.com/user-attachments/assets/686722cd-ccec-47d8-9cc9-3122ca555cc9" />)

*<p align="center">圖 1: 系統整體架構圖</p>*


## 環境建置與安裝 (Setup and Installation)

#### **先決條件 (Prerequisites)**

* Ubuntu 20.04
* ROS Noetic
* Python 3.8+
* PyTorch

#### **安裝步驟**

1.  **Clone 專案庫**:
    ```bash
    cd ~/catkin_ws/src
    git clone [你的專案 Git 連結]
    ```

2.  **安裝 Python 依賴套件**:
    ```bash
    cd [你的專案名稱]
    pip install -r requirements.txt
    ```

3.  **編譯 ROS 工作空間**:
    ```bash
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

## 如何運行 (How to Run)

#### **1. 啟動模擬環境中的巡邏任務**

```bash
# 啟動 Gazebo 模擬器與 Rviz
roslaunch [你的專案名稱] turtlebot3_world.launch

# 啟動已經訓練好的 Agent 來執行巡邏
roslaunch [你的專案名稱] patrol_simulation.launch model_name:=[你要載入的模型名稱]

