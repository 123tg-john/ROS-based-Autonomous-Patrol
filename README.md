# 基於深度強化學習之自主巡邏與異常偵測機器人 (Autonomous Patrol and Anomaly Detection Robot using DRL)

[![ROS Noetic](https://img.shields.io/badge/ROS-Noetic-blue)](http://wiki.ros.org/noetic)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange)](https://pytorch.org/)

本專案旨在開發一套能夠在複雜環境中，同時執行自主巡邏及即時異常偵測的智慧機器人系統。系統基於 **ROS (Robot Operating System)** 框架，並採用 **深度強化學習 (Deep Reinforcement Learning)** 中的 **Soft Actor-Critic (SAC)** 演算法進行模型訓練。

---

## 系統展示 (Demonstration)

[![專案展示影片](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID/0.jpg)](https://youtu.be/ZF6Ui-NO0R8)
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

[<img width="773" height="929" alt="模擬流程圖" src="https://github.com/user-attachments/assets/686722cd-ccec-47d8-9cc9-3122ca555cc9" />](https://github.com/123tg-john/ROS-based-Autonomous-Patrol/blob/main/assets/%E5%B0%88%E9%A1%8C%E5%9C%963.1.drawio.png?raw=true))

*<p align="center">圖 1: 系統整體架構圖</p>*


## 環境建置與安裝 (Setup and Installation)

#### **先決條件 (Prerequisites)**

* Ubuntu 20.04
* ROS Noetic
* Python 3.8+
* PyTorch

#### **初始環境設定**

在開始前，請依照以下三個核心步驟完成專案的下載與編譯。

1.  **Clone 專案庫**:
    ```bash
    cd ~/catkin_ws/src
    git clone [專案 Git 連結]
    ```

2.  **安裝 Python 依賴套件**:
    ```bash
    # 進入剛 Clone 下來的專案資料夾
    cd [你的專案名稱]
    pip install -r requirements.txt
    ```

3.  **編譯 ROS 工作空間**:
    ```bash
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

## 專案結構與檔案說明

本專案已將各主要功能的詳細說明，分別放置於對應的資料夾中。在您完成上述安裝步驟後，建議您瀏覽以下資料夾內的 `README.md` 檔案，以深入了解其內容與用途：

-   `launch/`: 包含所有用於啟動模擬或訓練的 `.launch` 檔案。
-   `scripts/`: 包含專案核心功能的 Python 腳本 (ROS 節點)。
-   `maps/` & `worlds/`: 包含 2D 地圖、3D 模型與 Gazebo 環境檔案。
-   `stage_1/` & `stage_2/`: 包含各階段訓練好的模型檔案。

## 如何運行模擬 (How to Run the Simulation)

在您完成專案的安裝與編譯後，要啟動完整的自主巡邏模擬任務非常簡單。整個流程主要分為兩個步驟：首先是確保 3D 地圖模型被 Gazebo 模擬器找得到，其次是執行我們提供的一鍵啟動 `launch` 檔。

首先，請確保本專案提供的 3D 地圖模型檔案已放置於正確的路徑。您需要將專案中的模型資料夾（例如 `blender_map_project` 整個資料夾）完整複製到您家目錄 (`Home`) 底下的 `.gazebo/models/` 資料夾中。請注意，`.gazebo` 是一個隱藏資料夾，在檔案總管中您可能需要按下 `Ctrl + H` 才能看見它。

完成模型檔案的放置後，您就可以透過 `roslaunch` 指令來執行完整的巡邏模擬任務了。請打開一個新的終端機，並執行以下指令：

```bash
# 此處以啟動「代價地圖法」的模擬為例
roslaunch [你的專案名稱] run_patrol_costmap.launch
