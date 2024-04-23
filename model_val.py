from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.yaml')

    validation_results = model.val(data='data/object/data/labels/data.yaml',
                               imgsz=640,
                               batch=16,
                               device='0')