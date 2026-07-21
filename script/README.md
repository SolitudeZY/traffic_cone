# 交通导改 OBB 训练与推理

当前主方案面向 RTX 5060 Ti，使用单个 `YOLOv8s-OBB` 四分类模型：`cone`、`traffic_sign`、`person`、`vest`。交通锥原始四点 OBB 保留不变，PPE 的 person/vest AABB 转成水平四点 OBB。OBB 四角可直接计算中心点和长轴角度，无需另外标中心点。

## 1. 环境

在项目根目录执行：

```bash
conda env create -f script/environment.yml
conda activate yolo-obb
```

## 2. 构建融合数据集

输入：

- `dataset/data-26-07-14/`：交通锥和交通标志 OBB 数据。
- `dataset/person/PPE-supplement/20260612/`：PPE AABB 数据，只使用 person 和 vest。

构建命令：

```bash
python script/build_fused_dataset.py
```

输出到 `dataset/2026-07-15/`：

- `images/{train,val,test}/`：源图片的硬链接，不重复占用图片磁盘空间。
- `labels/{train,val,test}/*.txt`：统一的四点 OBB 标签。
- `data.yaml`：Ultralytics 数据配置。
- `manifest.json`：源路径、类别映射、图片数和目标数。

训练集中的交通锥图片默认重复 3 份，以缓解 PPE 图片数量更多造成的类别偏置；验证集和测试集不重复。构建脚本不会修改源数据，并拒绝覆盖非空输出目录。

## 3. 数据校验与训练

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

`train_obb.py` 的默认参数已是上述融合数据、`YOLOv8s-OBB` 和 960 尺寸，因此也可直接执行 `python script/train_obb.py --device 0 --cache`。

训练输出在 `script/runs/traffic_cone_person_vest_yolov8s_obb_960/`：`weights/best.pt` 是验证集最优权重，`weights/last.pt` 是最后一轮检查点，`results.csv` 和 `results.png` 是训练曲线，混淆矩阵和验证预览图用于检查各类别效果。

断点续训：

```bash
python script/train_obb.py \
  --resume script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/last.pt \
  --device 0
```

## 4. 图片或视频推理

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 960 \
  --device 0
```

`--source` 可输入单张图片、图片目录、视频、`rtsp://...` 推流地址或摄像头编号 `0`。结果默认保存在 `script/runs/infer/`：图片输出为 `*_result.jpg`，视频输出为 `*_result.mp4`，逐帧结构化结果为 `events.jsonl`。

融合模型默认使用同一次前向中的 `person` 和 `vest` 框抑制人体区域内的假锥桶，不需要二阶段人员模型，也不会产生第二次完整推理的 FPS 开销。需要关闭时传入 `--exclude-classes ''`。

视频推理会先按当前帧生成空间 route，再进行 route 级时序跟踪。每个已确认的锥桶列拥有稳定 ID、独立丢失计数和经过平滑的 ROI；短暂漏检时显示 `AUTO ROI <id> HOLD`，重新连续匹配后恢复。历史 route 只能辅助拆分本帧已经连在一起的组，不能跨越本帧不连通的空间组进行合并。

推理默认根据锥桶最近邻间距生成多个自适应 ROI。相距较远的锥桶列会分别排序、连线和计算告警，不会跨列连接。`--route-link-ratio` 默认为 `3.0`；调小会更积极地拆分锥桶列，调大则允许连接更宽的缺失空档。仍可用像素 ROI 先限定需要分析的区域：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --roi 100,200,1800,1000 \
  --device 0
```

无人机巡航场景的时序参数默认值如下：

- `--route-confirm-frames 3`：新 route 连续出现 3 帧才确认，抑制单帧误检 ROI。
- `--route-recovery-frames 2`：保持中的 route 连续匹配 2 帧后恢复业务告警。
- `--roi-memory-frames 5`：每条已确认 route 独立保留 5 个漏检帧；其他 route 仍可见时不会重置它。
- `--route-match-ratio 2.5`：候选 route 与预测 route 的匹配距离上限，相对于局部锥距计算。
- `--route-max-angle 45`：匹配允许的 route 主方向最大差值，单位为度。
- `--roi-smoothing 0.35`：缓慢运动时的 ROI 观测权重，越小越平稳但跟随更慢。
- `--roi-max-smoothing 0.75`：快速运动或预测误差较大时的最大观测权重。
- `--no-temporal-roi`：关闭时序 route 跟踪，退回逐帧自适应 ROI，适合对照诊断。

每个稳定 ROI 还会独立记录缺失判断实际使用过的局部参考间距，防止两个靠得过近的锥桶把后续阈值拉低：

- `--gap-history-frames 30`：每个 ROI 最多保留 30 个候选告警帧的局部参考均值。
- `--gap-history-min-frames 3`：至少 3 个一致参考帧后才启用历史阈值；短 route 单帧出现时不会贸然告警。
- `--gap-close-ratio 0.55`：当前参考低于历史基准的 55% 时视为近距污染，不写入历史。
- `--gap-max-drop-ratio 0.05`：历史基准每帧最多降低 5%，既防止突降，也允许无人机飞高或变焦后逐步适应。
- `--gap-min-pixels 0`：仅旧像素模式使用的固定安全下限；透视校正模式不会把像素值混入地面距离。
- `--gap-vote-window 5` 和 `--gap-confirm-votes 3`：同一个稳定 gap 在最近 5 帧内至少 3 帧为候选，才发布 `MISSING`。
- `--gap-clear-frames 3`：已确认 gap 连续 3 帧正常后解除告警，避免阈值附近反复闪烁。
- `--gap-membership-cooldown 5`：锥桶进入或离开 route 后，相邻 gap 暂停 5 帧；参数只允许 3-5 帧。
- `--min-alert-cones 3`：当前 ROI 少于 3 个锥桶时不产生稳定业务告警。`--min-route-cones` 仍默认为 2，因此两锥桶 ROI 会继续画框展示。

历史基准使用“各帧局部参考均值的中位数”，不会永久取历史最大值。持续发生尺度变化时它可以下降；单帧或少量异常小值不会污染基准。近距锥桶仍保留在检测、ROI、偏移和倒伏分析中，只排除其对缺失参考阈值的影响。

例如，使用最终稳定配置推理两个测试视频：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 960 \
  --device 0 \
  --output script/runs/infer_videos_gap_stable/test_10s

python script/infer_obb.py \
  --weights script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/DJI_1m22s_1m36s.mp4 \
  --imgsz 960 \
  --device 0 \
  --output script/runs/infer_videos_gap_stable/DJI_1m22s_1m36s
```

输出目录包含标注视频 `*_result.mp4`、逐帧诊断 `events.jsonl`、结构化告警激活记录 `alert.jsonl` 和人工复查日志 `alert.log`。同一告警持续多帧时，后两者只在激活瞬间写一条，不会每帧重复。`alert.log` 示例：

```text
1\ttime=2.500s\troi=ROI_4\ttype=MISSING
```

时间是视频播放秒数，不是推理机器的系统时间。`events.jsonl` 中：

- `adaptive_routes` 是当前帧的原始空间分组，包含 `route_id`、`stable_route_id`、原始 ROI 和成员中心点。
- `tracked_routes` 是时序结果，包含稳定 ID、`tentative/confirmed/held` 状态、平滑 ROI、原始 ROI、速度、命中数、丢失帧数和 `alert_ready`。
- `gap_thresholds` 按稳定 ROI 输出历史基准、当前帧均值、历史均值、距离单位、置信度、被排除的近距样本和阈值来源；透视模式使用 `*_corrected` 字段，旧像素模式使用 `*_pixels` 字段。
- `tracked_gaps` 包含稳定 gap ID、3/5 投票数、成员变化冷却、连续正常帧数和业务资格，供程序诊断。
- `missing`、`displaced`、`fallen` 保留原始逐帧告警，兼容既有分析流程。
- `stable_missing`、`stable_displaced`、`stable_fallen` 只包含时序状态允许上报的告警。线上业务应优先消费这些字段，`tentative`、`held` 和恢复确认中的 route 不会进入稳定告警。

包含近距污染保护的最新测试输出位于 `script/runs/infer_videos_gap_stable/test_10s/` 和 `script/runs/infer_videos_gap_stable/DJI_1m22s_1m36s/`。

旧的两分类锥桶权重仍可选用外置 person 模型抑制反光服假锥桶：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 960 \
  --person-weights yolo26n.pt \
  --person-classes 0 \
  --device 0
```

如使用项目 PPE 权重，person 和 vest 类为 `0,2`；当前测试视频中的边缘截断人员仍建议使用通用 `yolo26n.pt`。

规则默认值：

- 自适应 ROI：中心距离不超过局部最近邻间距的 `3.0` 倍时归入同一序列。
- 缺失：每个序列内局部相邻间距大于邻域中位数的 `1.8` 倍。
- 偏移：有标定时超过 `0.5m`；未标定时超过中位锥距的 `25%`。
- 倒伏：完整 OBB 长轴相对画面竖直方向偏转至少 `30` 度。
- 边缘保护：画面边缘 `5%` 内且角度失真的框标为 `EDGE SKIP`，不参与空间规则。

无人机推理默认把相机视为针孔相机，将锥桶中心射线与水平地面求交。默认光轴相对水平面向下 `45°`，水平视场角 `84°`：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/DJI_1m22s_1m36s.mp4 \
  --camera-pitch-deg 45 \
  --camera-hfov-deg 84
```

未提供高度时，校正距离单位为 `camera_heights`（相机离地高度的倍数），足以进行缺失间距比例和相对偏移判断，但不是米。提供相机离锥桶所在平面的垂直高度后才输出米制距离：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/DJI_1m22s_1m36s.mp4 \
  --camera-pitch-deg 45 \
  --camera-hfov-deg 84 \
  --drone-altitude-m 30
```

`events.jsonl` 同时保留 `gap_pixels`/`distance_pixels` 和 `gap_corrected`/`distance_corrected`，并在 `perspective_calibration` 中记录俯角、视场角、高度、像素焦距和单位。

旧的局部像素标定仍可用于小范围固定机位；显式传入 `--meters-per-pixel` 时会使用旧模式而不叠加透视校正。例如地面上已知长度为 10m 的线段在图中为 250px：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --meters-per-pixel 0.04
```

无人机画面需保持云台滚转稳定。默认校正假设锥桶位于同一水平地面；地面有明显坡度、相机存在滚转、画面被裁切或实际视场角未知时，应使用实测内外参和地面单应性矩阵。可用 `--no-perspective-correction` 显式退回旧像素距离模式。

## 5. 独立 PPE/person 抑制模型（兼容旧权重）

数据位于 `dataset/person/PPE-supplement/20260612`。训练脚本会自动把源 YAML 中失效的 `/mnt/data/...` 改写到运行时配置，不修改原 YAML。

```bash
python script/train_ppe.py \
  --device 0 \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --name ppe_person_detector
```

最佳权重为 `script/runs/ppe_person_detector/weights/best.pt`。使用方式：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 960 \
  --person-weights script/runs/ppe_person_detector/weights/best.pt \
  --person-classes 0,2 \
  --device 0
```

## 数据注意事项

融合数据集共 1021 张训练图、231 张验证图、153 张测试图。交通标志牌全部划分合计仍只有 18 个目标（训练重复后统计），样本远不足以可靠训练；现阶段模型评估应以锥桶、人员和反光服为主，后续需要补充标志牌的正面、侧面、遮挡、尺度和光照样本。
