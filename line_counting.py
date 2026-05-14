
import os
import cv2
import numpy as np
from ultralytics import YOLO


def main():
    model_path = os.path.join(os.path.dirname(__file__), "runs", "detect", "runs", "detect", "vehicle_detection", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train.py 完成训练")
        return

    video_path = os.path.join(os.path.dirname(__file__), "路口视频.mp4")
    output_path = os.path.join(os.path.dirname(__file__), "counting_output.mp4")

    model = YOLO(model_path)

    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 设定虚拟计数线（水平线，位于画面中间偏下位置）
    line_y = int(height * 0.6)
    line_start = (0, line_y)
    line_end = (width, line_y)

    print(f"视频信息: {width}x{height}, {fps}fps, 共{total_frames}帧")
    print(f"计数线位置: y={line_y} (画面60%高度处)")
    print(f"开始越线计数...")

    # 跟踪每个ID的历史中心y坐标
    track_history = {}  # {track_id: previous_cy}
    crossed_ids = set()  # 已经越线的ID集合
    cross_count = 0

    # 使用 YOLOv8 跟踪
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    cap = cv2.VideoCapture(video_path)

    for result in results:
        ret, frame = cap.read()
        if not ret:
            break

        # 绘制检测框和跟踪ID
        annotated = result.plot()

        # 绘制计数线
        cv2.line(annotated, line_start, line_end, (0, 0, 255), 2)

        # 越线检测逻辑
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for i, tid in enumerate(track_ids):
                x1, y1, x2, y2 = boxes[i]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # 绘制中心点
                cv2.circle(annotated, (cx, cy), 3, (0, 255, 255), -1)

                # 判断是否越线
                if tid in track_history and tid not in crossed_ids:
                    prev_cy = track_history[tid]
                    # 从上往下越线 或 从下往上越线
                    if (prev_cy < line_y and cy >= line_y) or (prev_cy > line_y and cy <= line_y):
                        crossed_ids.add(tid)
                        cross_count += 1

                track_history[tid] = cy

        # 绘制计数信息
        cv2.rectangle(annotated, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.putText(annotated, f"Crossed: {cross_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(annotated, "Counting Line", (width - 200, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        writer.write(annotated)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  进度: {frame_count}/{total_frames} 帧, 当前越线数: {cross_count}")

    cap.release()
    writer.release()

    print(f"\n越线计数完成!")
    print(f"总越线数: {cross_count}")
    print(f"输出视频: {output_path}")


if __name__ == "__main__":
    main()
