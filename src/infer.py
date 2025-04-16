from ultralytics import YOLO
import cv2, os

def inference(model, img, output_path):
    results = model(img)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].xyxy[0]
            class_id = int(class_ids[i])
            score = scores[i]

            color = (255, 0, 0) if class_id == 1 else (0, 0, 255)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"{class_id} {score:.2f}"
            cv2.putText(img, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(output_path, img)


if __name__ == "__main__":
    # metadata_model = YOLO("../models/metadata_yolov8m/weights/best.pt")
    # metadata_input_img_dir = "../dataset1/metadata/valid/images"
    # metadata_output_img_dir = "../validate_result/metadata_model"
    # for img_name in os.listdir(metadata_input_img_dir):
    #     img_path = os.path.join(metadata_input_img_dir, img_name)
    #     img = cv2.imread(img_path)
    #     output_path = os.path.join(metadata_output_img_dir, img_name)
    #     inference(metadata_model, img, output_path)
    #     print(f"Processed {img_name} with metadata model.")

    content_model = YOLO("../models/content_yolov8m/weights/best.pt")
    content_input_img_dir = "../dataset1/content/valid/images"
    content_output_img_dir = "../validate_result/content_model"
    os.makedirs(content_output_img_dir, exist_ok=True)
    for img_name in os.listdir(content_input_img_dir):
        img_path = os.path.join(content_input_img_dir, img_name)
        img = cv2.imread(img_path)
        output_path = os.path.join(content_output_img_dir, img_name)
        inference(content_model, img, output_path)
        print(f"Processed {img_name} with content model.")
    