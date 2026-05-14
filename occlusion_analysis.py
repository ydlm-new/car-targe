
import os
import cv2
import numpy as np
from ultralytics import YOLO


def compute_iou(box1, box2):
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def find_occlusion_frames(results_list):
    """找到发生遮挡（高IoU重叠）的帧"""
    occlusion_scores = []

    for frame_idx, result in enumerate(results_list):
        if result.boxes is None or len(result.boxes) < 2:
            occlusion_scores.append(0)
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        max_iou = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = compute_iou(boxes[i], boxes[j])
                max_iou = max(max_iou, iou)

        occlusion_scores.append(max_iou)

    return occlusion_scores


def draw_tracking_frame(frame, result, frame_idx):
    """在帧上绘制跟踪结果，突出显示ID"""
    annotated = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None

    colors = {}
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        tid = track_ids[i] if track_ids is not None else -1

        if tid not in colors:
            np.random.seed(tid * 7 + 13)
            colors[tid] = tuple(int(c) for c in np.random.randint(50, 255, 3))

        color = colors[tid]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        cls_name = result.names[classes[i]]
        label = f"ID:{tid} {cls_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 绘制中心点
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 4, color, -1)

    # 帧号标注
    cv2.putText(annotated, f"Frame #{frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotated


def main():
    model_path = os.path.join(os.path.dirname(__file__), "runs", "detect", "runs", "detect", "vehicle_detection", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train.py 完成训练")
        return

    video_path = os.path.join(os.path.dirname(__file__), "路口视频.mp4")
    output_dir = os.path.join(os.path.dirname(__file__), "occlusion_frames")
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    print("第一遍：扫描视频寻找遮挡帧...")
    results_list = []
    results = model.track(
        source=video_path,
        tracker="botsort.yaml",
        conf=0.3,
        iou=0.5,
        show=False,
        stream=True,
        device="cpu",
        verbose=False,
    )

    # 收集所有帧的跟踪结果
    cap = cv2.VideoCapture(video_path)
    frames = []
    for result in results:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        results_list.append(result)
    cap.release()

    # 计算遮挡分数
    occlusion_scores = find_occlusion_frames(results_list)

    # 找到遮挡最严重的连续4帧
    best_start = 0
    best_score = 0
    for i in range(len(occlusion_scores) - 3):
        score = sum(occlusion_scores[i:i + 4])
        if score > best_score:
            best_score = score
            best_start = i

    print(f"找到遮挡最严重的片段: 帧 {best_start} ~ {best_start + 3}")
    print(f"遮挡分数: {[f'{s:.3f}' for s in occlusion_scores[best_start:best_start+4]]}")

    # 重新跟踪并保存这4帧
    print("\n第二遍：重新跟踪并保存遮挡帧可视化...")

    # 直接使用第一遍的结果
    for i in range(4):
        idx = best_start + i
        if idx < len(frames):
            annotated = draw_tracking_frame(frames[idx], results_list[idx], idx)
            save_path = os.path.join(output_dir, f"occlusion_frame_{i+1}_idx{idx}.png")
            cv2.imwrite(save_path, annotated)
            print(f"  保存: {save_path}")

    # 分析ID变化
    print("\n=== 遮挡分析报告 ===")
    print(f"分析帧范围: {best_start} ~ {best_start + 3}")

    all_ids_per_frame = []
    for i in range(4):
        idx = best_start + i
        if idx < len(results_list):
            result = results_list[idx]
            if result.boxes.id is not None:
                ids = set(result.boxes.id.cpu().numpy().astype(int).tolist())
            else:
                ids = set()
            all_ids_per_frame.append(ids)
            print(f"  帧 {idx}: 检测到 {len(ids)} 个目标, IDs = {sorted(ids)}")

    # 检测ID跳变
    if len(all_ids_per_frame) >= 2:
        print("\nID变化分析:")
        for i in range(1, len(all_ids_per_frame)):
            lost = all_ids_per_frame[i-1] - all_ids_per_frame[i]
            new = all_ids_per_frame[i] - all_ids_per_frame[i-1]
            if lost:
                print(f"  帧{best_start+i-1}→帧{best_start+i}: 丢失ID {sorted(lost)}")
            if new:
                print(f"  帧{best_start+i-1}→帧{best_start+i}: 新增ID {sorted(new)} (可能为ID跳变)")
            if not lost and not new:
                print(f"  帧{best_start+i-1}→帧{best_start+i}: ID保持稳定")

    print(f"\n可视化结果已保存到: {output_dir}/")


if __name__ == "__main__":
    main()
