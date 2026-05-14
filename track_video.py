
import os
import cv2
from ultralytics import YOLO


def main():
    # 模型路径（训练完成后的最佳模型）
    model_path = os.path.join(os.path.dirname(__file__), "runs", "detect", "runs", "detect", "vehicle_detection", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train.py 完成训练")
        return

    # 视频路径
    video_path = os.path.join(os.path.dirname(__file__), "路口视频.mp4")
    output_path = os.path.join(os.path.dirname(__file__), "tracking_output.mp4")

    # 加载模型
    model = YOLO(model_path)

    # 打开视频获取信息
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"视频信息: {width}x{height}, {fps}fps, 共{total_frames}帧")
    print(f"开始多目标跟踪...")

    # 使用 YOLOv8 内置跟踪（BoT-SORT）
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

    # 写入输出视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    for result in results:
        frame = result.plot()
        writer.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  处理进度: {frame_count}/{total_frames} 帧")

    writer.release()
    print(f"\n跟踪完成! 输出视频: {output_path}")
    print(f"共处理 {frame_count} 帧")


if __name__ == "__main__":
    main()
