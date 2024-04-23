from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.yaml')

    results = model.train(data=r"D:\pycharmproject\ultralytics-main\data\object\data\labels\data.yaml",
                          resume=True,
                          epochs=100,
                          imgsz=640)

