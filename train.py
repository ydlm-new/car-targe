
import os
import swanlab
from ultralytics import YOLO


def on_train_epoch_end(trainer):
    """每个训练epoch结束时记录指标"""
    metrics = trainer.metrics
    epoch = trainer.epoch
    # 记录训练loss
    swanlab.log({
        "train/box_loss": trainer.loss_items[0].item(),
        "train/cls_loss": trainer.loss_items[1].item(),
        "train/dfl_loss": trainer.loss_items[2].item(),
        "train/total_loss": trainer.loss.item(),
        "lr/pg0": trainer.optimizer.param_groups[0]["lr"],
    }, step=epoch)


def on_val_end(validator):
    """验证结束时记录指标"""
    metrics = validator.metrics
    epoch = validator.training_epoch if hasattr(validator, 'training_epoch') else 0
    swanlab.log({
        "val/box_loss": validator.loss[0].item() if hasattr(validator, 'loss') else 0,
        "val/cls_loss": validator.loss[1].item() if hasattr(validator, 'loss') else 0,
        "val/dfl_loss": validator.loss[2].item() if hasattr(validator, 'loss') else 0,
        "val/mAP50": metrics.results_dict.get("metrics/mAP50(B)", 0),
        "val/mAP50-95": metrics.results_dict.get("metrics/mAP50-95(B)", 0),
        "val/precision": metrics.results_dict.get("metrics/precision(B)", 0),
        "val/recall": metrics.results_dict.get("metrics/recall(B)", 0),
    })


def on_train_end(trainer):
    """训练结束时关闭SwanLab"""
    swanlab.finish()


def main():
    # 初始化 SwanLab（离线模式，无需登录）
    swanlab.init(
        project="YOLOv8-Vehicle-Detection",
        experiment_name="yolov8n-road-vehicle",
        mode="local",
        config={
            "model": "yolov8n",
            "dataset": "Road Vehicle Images Dataset",
            "epochs": 30,
            "batch_size": 16,
            "imgsz": 640,
            "optimizer": "SGD",
            "lr0": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "num_classes": 21,
            "train_images": 2704,
            "val_images": 300,
        }
    )

    # 加载预训练模型
    model = YOLO("yolov8n.pt")

    # 添加回调
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_train_end", on_train_end)

    # 数据集配置路径
    data_yaml = os.path.join(os.path.dirname(__file__), "trafic_data", "data.yaml")

    # 开始训练
    results = model.train(
        data=data_yaml,
        epochs=30,
        batch=16,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        device="cpu",
        workers=4,
        project="runs/detect",
        name="vehicle_detection",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    print("\n训练完成!")
    print(f"最佳模型保存在: runs/detect/vehicle_detection/weights/best.pt")
    print(f"训练结果: mAP50={results.results_dict.get('metrics/mAP50(B)', 0):.4f}, "
          f"mAP50-95={results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")


if __name__ == "__main__":
    main()
