# Traffic Cone & PPE Detection (OBB)

这是一个基于 YOLOv8s-OBB (Oriented Bounding Box) 的交通导改锥桶与人员/反光服检测项目，专为无人机巡航、临边安全防护及环水保监测等场景设计。项目集成了空间路线自适应分组（Route Tracking）、局部间距平滑（Gap Tracking）、时序状态判定与告警过滤逻辑。

## 功能特性

- **多类别 OBB 检测**：支持 `cone` (锥桶)、`traffic_sign` (交通标志)、`person` (人员) 和 `vest` (反光服) 四个类别的旋转框检测。
- **空间路线分组 (Route Grouping)**：自动将画面中符合逻辑距离的锥桶聚合成连续路径 (Route/ROI)。
- **时序跟踪与防抖**：采用多帧投票 (Voting Window) 和状态机，避免由于无人机视角变化导致的锥桶短暂漏检或抖动。
- **透视畸变校正 (Perspective Correction)**：支持设定无人机摄像头俯仰角、视场角和飞行高度，在针孔相机模型下将像素距离转换为物理地面距离 (米)。
- **结构化告警日志**：区分逐帧检测结果与经过业务防抖过滤的稳态告警，自动输出可视化视频、`events.jsonl` (详细)、`alert.jsonl` (结构化告警) 和 `alert.log` (人工复查日志)。

## 环境安装

请在项目根目录执行以下命令，使用 Conda 创建并激活所需环境：

```bash
conda env create -f script/environment.yml
conda activate yolo-obb
```

## 数据集处理与训练

### 构建融合数据集

如果需要混合 OBB 锥桶数据和 AABB 人员数据，可以使用自带脚本将人员水平框转换为四点 OBB：

```bash
python script/build_fused_dataset.py
```
> 数据集默认位于 `dataset/2026-07-15/`，生成的配置文件为 `data.yaml`。

### 训练模型

验证数据集后即可开始训练，以下提供推荐参数 (当前主力模型采用 `YOLOv8s-OBB`，输入尺寸 960)：

```bash
python script/validate_dataset.py --data-root dataset/2026-07-15
python script/train_obb.py \
  --data dataset/2026-07-15/data.yaml \
  --model yolov8s-obb.pt \
  --device 0 \
  --epochs 150 \
  --batch 16 \
  --imgsz 960 \
  --cache \
  --name traffic_cone_person_vest_yolov8s_obb_960
```
> 若意外中断，可通过 `--resume script/runs/.../weights/last.pt` 恢复训练。

## 视频/图像推理与跟踪

### 基础视频推理

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 960 \
  --device 0
```
结果默认保存在 `script/runs/infer/` 目录下。

### 无人机透视校正推理 (推荐)

无人机推理默认把相机视为针孔相机。指定俯角、视场角和离地高度，可以得到准确的物理告警判定（以米为单位）：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/DJI_1m22s_1m36s.mp4 \
  --imgsz 960 \
  --device 0 \
  --camera-pitch-deg 45 \
  --camera-hfov-deg 84 \
  --drone-altitude-m 30 \
  --output script/runs/infer_videos_perspective45/DJI_1m22s_1m36s
```

## 日志与输出格式

推理结束后，输出目录中会包含以下内容：
- `*_result.mp4`: 渲染了 OBB 边框、Route ROI 区域及告警文本的视频。
- `events.jsonl`: 包含每一帧的检测框、空间分组原始结果、相机外参、时序跟踪细节。
- `alert.jsonl`: 经过时序投票（默认 3/5 窗口）与成员变化冷却过滤后激活的稳态告警，不会每帧重复，只在状态跃迁时记录。
- `alert.log`: 供业务人员直接阅读的纯文本日志，包含发生时间与告警类型（如 `MISSING`、`SHIFT`、`FALLEN`）。

## 核心规则与业务逻辑约束

- **路线识别下限**：有效 ROI 必须包含至少 3 个锥桶才会产生稳态的业务告警，不足 3 个（比如只有 2 个锥桶在画面边缘）会持续绘制 ROI 但抑制误报。
- **动态基准 (Adaptive Reference)**：针对曲线或锥桶摆放不均的场景，系统采用局部最近邻间距的平滑基准，避免全局单一阈值带来的漏报。
- **近距污染保护**：针对无人机飞近、高度下降引起的画面锥桶突然放大，动态阈值做了上限与异常过滤机制，保障检测的稳定性。
