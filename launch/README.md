# 啟動檔 (Launch Files)

這個資料夾存放了所有用於啟動本專案相關節點 (Node) 的 `.launch` 檔案。`.launch` 檔案是 ROS 中用來一次性啟動並配置多個節點的 XML 腳本，是執行本專案的入口。

## 主要啟動檔

本專案主要包含以下兩份用於**功能展示**的核心啟動檔：

1.  **`run_patrol_costmap.launch`**:
    * **用途**：啟動使用「**代價地圖法**」模型的完整模擬與部署環境。
    * **功能**：此檔案會啟動 Gazebo、RViz、`map_server`、`amcl`，並運行 `patrol_manager_costmap.py` 腳本。

2.  **`run_patrol_scan_diff.launch`**:
    * **用途**：啟動使用「**基線比對法**」模型的完整模擬與部署環境。
    * **功能**：此檔案會啟動必要的模擬節點，並運行 `patrol_manager_scan_diff.py` 腳本。

除了功能展示檔外，資料夾中可能還包含用於**模型訓練**的 `launch` 檔（例如 `train_s2_costmap.launch` 等）。

## 如何使用

在您的 `catkin_ws` 工作空間中，使用 `roslaunch` 指令來執行這些檔案。

```bash
# 範例：啟動使用「代價地圖法」的模擬巡邏任務
roslaunch your_package_name run_patrol_costmap.launch
