# 地圖與模擬環境檔案 (Maps and Simulation Worlds)

這個資料夾存放了本專案中用於 ROS 模擬與視覺化的所有地圖相關檔案。

## 檔案類型與用途

### 2D 地圖檔 (.png & .yaml)

這兩個檔案是 ROS Navigation Stack 中 `map_server` 節點使用的標準地圖格式，主要用於 **RViz 中的 2D 地圖視覺化**與 **AMCL 定位**。

* **`.png` 檔案**：這是地圖的影像檔，是一張灰階圖片。
    * **黑色**像素：代表牆壁等障礙物。
    * **白色**像素：代表可通行的區域。
    * **灰色**像素：代表未知空間。

* **`.yaml` 檔案**：這是地圖的設定檔，用來描述對應的 `.png` 影像檔的屬性，例如解析度 (`resolution`)、原點 (`origin`) 等重要資訊，讓 ROS 知道如何將圖片像素對應到真實世界的座標。

#### **如何使用 (.png & .yaml)**
這兩個檔案通常會放在 `maps` 資料夾中。在啟動 `launch` 檔案時，`map_server` 節點會被指定載入這個 `.yaml` 檔，隨後它會自動找到對應的 `.png` 檔並將地圖發布到 `/map` 主題上，RViz 便能訂閱並顯示出來。

*Launch 檔案中的範例：*
```xml
<node name="map_server" pkg="map_server" type="map_server" args="$(find your_package_name)/maps/your_map.yaml" />
