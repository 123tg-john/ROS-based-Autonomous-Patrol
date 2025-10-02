# 地圖與模擬環境檔案 (Maps and Simulation Worlds)

這個資料夾存放了本專案中用於 Gazebo 模擬的所有地圖相關檔案，主要包含 3D 模型檔 (`.dae`) 與 Gazebo 世界檔 (`.world`)。

## 檔案來源與生成方式

本專案中的 3D 模型 (`.dae` 檔) 皆源自於**實體機器人透過 SLAM 技術建立的 2D 環境地圖**。我們將 2D 地圖匯入至 Blender 3D 建模軟體中，再透過以下兩種方式將其轉換為 3D 模型：

1.  **矩形地圖 (Rectangular Map)**
    * **方法**：採用手動方式，參照 2D 地圖的佈局，在 Blender 中放置標準的立方體方塊來建構牆體。
    * **特色**：建立出一個牆體筆直、轉角規律的理想化、結構化室內環境。

2.  **點陣地圖 (Occupancy Grid Map)**
    * **方法**：利用客製化的 Python 腳本，在 Blender 中自動讀取 2D 地圖的 `.png` 圖片檔。腳本會將圖片中的黑色像素（代表牆壁）辨識出來，並自動生成對應的 3D 網格模型。
    * **特色**：建立出的環境邊緣帶有不規則紋理，更貼近真實 SLAM 掃描時的感測器雜訊，用於測試模型的強健性。

## 如何使用

### 3D 模型檔 (`.dae`)

為了讓 Gazebo 能夠找到並載入這些 3D 環境模型，您必須將 `.dae` 檔案以及其對應的 `model.config` 和 `model.sdf` 等設定檔，放置到 Gazebo 的模型資料庫路徑中。

1.  打開您的 Linux 檔案總管。
2.  前往您的家目錄 (`/home/你的使用者名稱/`)。
3.  按下 **`Ctrl + H`** 來顯示隱藏的檔案與資料夾。
4.  找到並進入 `.gazebo/models/` 資料夾。
5.  將本專案提供的模型資料夾完整複製到此處。

### Gazebo 世界檔 (`.world`)

`.world` 檔案是用於 `roslaunch` 指令中，可以直接啟動一個包含特定地圖模型、光照和其他物理設定的 Gazebo 模擬環境。您通常會在專案的 `launch` 資料夾中看到它們被引用。

```xml
<include file="$(find gazebo_ros)/launch/empty_world.launch">
  <arg name="world_name" value="$(find your_package_name)/worlds/your_map.world"/>
</include>
