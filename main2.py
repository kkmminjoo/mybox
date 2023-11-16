import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
from PIL import Image
import io

app = FastAPI()

def yolo_multitask_loss(y_true, y_pred):

    batch_loss = 0
    epsilon = 1e-6
    count = len(y_true)
    for i in range(0, len(y_true)) :
        y_true_unit = tf.identity(y_true[i])
        y_pred_unit = tf.identity(y_pred[i])

        y_true_unit = tf.reshape(y_true_unit, [49, 12])
        y_pred_unit = tf.reshape(y_pred_unit, [49, 17])

        loss = 0

        for j in range(0, len(y_true_unit)) :

            bbox1_pred = tf.identity(y_pred_unit[j][:4])
            bbox1_pred_confidence = tf.identity(y_pred_unit[j][4])
            bbox2_pred = tf.identity(y_pred_unit[j][5:9])
            bbox2_pred_confidence = tf.identity(y_pred_unit[j][9])
            class_pred = tf.identity(y_pred_unit[j][10:])

            bbox_true = tf.identity(y_true_unit[j][:4])
            bbox_true_confidence = tf.identity(y_true_unit[j][4])
            class_true = tf.identity(y_true_unit[j][5:])

            bbox1_pred = tf.maximum(0.0, bbox1_pred)
            bbox2_pred = tf.maximum(0.0, bbox2_pred)

            box_pred_1_np = bbox1_pred.numpy()
            box_pred_2_np = bbox2_pred.numpy()
            box_true_np   = bbox_true.numpy()

            box_pred_1_area = box_pred_1_np[2] * box_pred_1_np[3]
            box_pred_2_area = box_pred_2_np[2] * box_pred_2_np[3]
            box_true_area   = box_true_np[2]  * box_true_np[3]


            box_pred_1_minmax = np.asarray([box_pred_1_np[0] - 0.5*box_pred_1_np[2], box_pred_1_np[1] - 0.5*box_pred_1_np[3], box_pred_1_np[0] + 0.5*box_pred_1_np[2], box_pred_1_np[1] + 0.5*box_pred_1_np[3]])
            box_pred_2_minmax = np.asarray([box_pred_2_np[0] - 0.5*box_pred_2_np[2], box_pred_2_np[1] - 0.5*box_pred_2_np[3], box_pred_2_np[0] + 0.5*box_pred_2_np[2], box_pred_2_np[1] + 0.5*box_pred_2_np[3]])
            box_true_minmax   = np.asarray([box_true_np[0] - 0.5*box_true_np[2], box_true_np[1] - 0.5*box_true_np[3], box_true_np[0] + 0.5*box_true_np[2], box_true_np[1] + 0.5*box_true_np[3]])

            InterSection_pred_1_with_true = [max(box_pred_1_minmax[0], box_true_minmax[0]), max(box_pred_1_minmax[1], box_true_minmax[1]), min(box_pred_1_minmax[2], box_true_minmax[2]), min(box_pred_1_minmax[3], box_true_minmax[3])]
            InterSection_pred_2_with_true = [max(box_pred_2_minmax[0], box_true_minmax[0]), max(box_pred_2_minmax[1], box_true_minmax[1]), min(box_pred_2_minmax[2], box_true_minmax[2]), min(box_pred_2_minmax[3], box_true_minmax[3])]

            IntersectionArea_pred_1_true = 0

            if (InterSection_pred_1_with_true[2] - InterSection_pred_1_with_true[0] + 1) >= 0 and (InterSection_pred_1_with_true[3] - InterSection_pred_1_with_true[1] + 1) >= 0 :
                    IntersectionArea_pred_1_true = (InterSection_pred_1_with_true[2] - InterSection_pred_1_with_true[0] + 1) * InterSection_pred_1_with_true[3] - InterSection_pred_1_with_true[1] + 1

            IntersectionArea_pred_2_true = 0

            if (InterSection_pred_2_with_true[2] - InterSection_pred_2_with_true[0] + 1) >= 0 and (InterSection_pred_2_with_true[3] - InterSection_pred_2_with_true[1] + 1) >= 0 :
                    IntersectionArea_pred_2_true = (InterSection_pred_2_with_true[2] - InterSection_pred_2_with_true[0] + 1) * InterSection_pred_2_with_true[3] - InterSection_pred_2_with_true[1] + 1

            Union_pred_1_true = box_pred_1_area + box_true_area - IntersectionArea_pred_1_true
            Union_pred_2_true = box_pred_2_area + box_true_area - IntersectionArea_pred_2_true

            IoU_box_1 = IntersectionArea_pred_1_true / (Union_pred_1_true + epsilon)
            IoU_box_2 = IntersectionArea_pred_2_true / (Union_pred_2_true + epsilon)

            responsible_IoU = 0
            responsible_box = 0
            responsible_bbox_confidence = 0
            non_responsible_bbox_confidence = 0

            if IoU_box_1 >= IoU_box_2 :
                responsible_IoU = IoU_box_1
                responsible_box = tf.identity(bbox1_pred)
                responsible_bbox_confidence = tf.identity(bbox1_pred_confidence)
                non_responsible_bbox_confidence = tf.identity(bbox2_pred_confidence)

            else :
                responsible_IoU = IoU_box_2
                responsible_box = tf.identity(bbox2_pred)
                responsible_bbox_confidence = tf.identity(bbox2_pred_confidence)
                non_responsible_bbox_confidence = tf.identity(bbox1_pred_confidence)

            obj_exist = tf.ones_like(bbox_true_confidence)
            if box_true_np[0] == 0.0 and box_true_np[1] == 0.0 and box_true_np[2] == 0.0 and box_true_np[3] == 0.0 :
                obj_exist = tf.zeros_like(bbox_true_confidence)

            localization_err_x = tf.math.pow( tf.math.subtract(bbox_true[0], responsible_box[0]), 2)
            localization_err_y = tf.math.pow( tf.math.subtract(bbox_true[1], responsible_box[1]), 2)

            localization_err_w = tf.math.pow( tf.math.subtract(tf.sqrt(bbox_true[2]), tf.sqrt(responsible_box[2])), 2)
            localization_err_h = tf.math.pow( tf.math.subtract(tf.sqrt(bbox_true[3]), tf.sqrt(responsible_box[3])), 2)

            if tf.math.is_nan(localization_err_w).numpy() == True :
                localization_err_w = tf.zeros_like(localization_err_w, dtype=tf.float32)

            if tf.math.is_nan(localization_err_h).numpy() == True :
                localization_err_h = tf.zeros_like(localization_err_h, dtype=tf.float32)

            localization_err_1 = tf.math.add(localization_err_x, localization_err_y)
            localization_err_2 = tf.math.add(localization_err_w, localization_err_h)
            localization_err = tf.math.add(localization_err_1, localization_err_2)

            weighted_localization_err = tf.math.multiply(localization_err, 5.0)
            weighted_localization_err = tf.math.multiply(weighted_localization_err, obj_exist)

            class_confidence_score_obj = tf.math.pow(tf.math.subtract(responsible_bbox_confidence, bbox_true_confidence), 2)
            class_confidence_score_noobj = tf.math.pow(tf.math.subtract(non_responsible_bbox_confidence, tf.zeros_like(bbox_true_confidence)), 2)
            class_confidence_score_noobj = tf.math.multiply(class_confidence_score_noobj, 0.5)

            class_confidence_score_obj = tf.math.multiply(class_confidence_score_obj, obj_exist)
            class_confidence_score_noobj = tf.math.multiply(class_confidence_score_noobj, tf.math.subtract(tf.ones_like(obj_exist), obj_exist)) # 객체가 존재하면 0, 존재하지 않으면 1을 곱합

            class_confidence_score = tf.math.add(class_confidence_score_obj,  class_confidence_score_noobj)

            classification_err = tf.math.pow(tf.math.subtract(class_true, class_pred), 2.0)
            classification_err = tf.math.reduce_sum(classification_err)
            classification_err = tf.math.multiply(classification_err, obj_exist)

            loss_OneCell_1 = tf.math.add(weighted_localization_err, class_confidence_score)
            loss_OneCell = tf.math.add(loss_OneCell_1, classification_err)

            if loss == 0 :
                loss = tf.identity(loss_OneCell)
            else :
                loss = tf.math.add(loss, loss_OneCell)

        if batch_loss == 0 :
            batch_loss = tf.identity(loss)
        else :
            batch_loss = tf.math.add(batch_loss, loss)

    count = tf.Variable(float(count))
    batch_loss = tf.math.divide(batch_loss, count)

    return batch_loss


model = load_model("cnn333.h5")
YOLO = load_model("yololasty_model.h5")

classes_yolo = ['biodegradable', 'cardboard', 'garbage', 'glass', 'metal', 'paper', 'plastic']

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    intersect_min_x = max(x1_min, x2_min)
    intersect_min_y = max(y1_min, y2_min)
    intersect_max_x = min(x1_max, x2_max)
    intersect_max_y = min(y1_max, y2_max)

    intersect_area = max(0, intersect_max_x - intersect_min_x) * max(0, intersect_max_y - intersect_min_y)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - intersect_area
    iou = intersect_area / union_area if union_area != 0 else 0

    return iou

def class_cut(bbox_list) :
    nms_bbox_list = []
    for i in range(0, len(bbox_list)) :

        if bbox_list[i][4] > 0.2:
          if bbox_list[i][5] != 'glass':
              nms_bbox_list.append(bbox_list[i])


    return nms_bbox_list

def nms(bbox_list, iou_threshold=0.2):
    if not bbox_list:
        return []

    bbox_list = class_cut(bbox_list)

    bbox_list = sorted(bbox_list, key=lambda x: x[4], reverse=True)
    nms_bbox_list = []

    while bbox_list:
        chosen_box = bbox_list.pop(0)
        nms_bbox_list.append(chosen_box)
        bbox_list = [box for box in bbox_list if calculate_iou(chosen_box[:4], box[:4]) < iou_threshold]

    return nms_bbox_list

def process_bbox(x, y, bbox, image_size, classes_score, Classes_inDataSet):
    grid_size = 32.0

    bbox_x_center = (x + bbox[0]) * grid_size * (image_size[0] / 224.0)
    bbox_y_center = (y + bbox[1]) * grid_size * (image_size[1] / 224.0)
    bbox_w = bbox[2] * image_size[0]
    bbox_h = bbox[3] * image_size[1]

    min_x = int(bbox_x_center - bbox_w / 4)
    min_y = int(bbox_y_center - bbox_h / 4)
    max_x = int(bbox_x_center + bbox_w / 4)
    max_y = int(bbox_y_center + bbox_h / 4)

    idx_class_highest_score = np.argmax(classes_score)
    class_highest_score = classes_score[idx_class_highest_score]
    class_highest_score_name = Classes_inDataSet[idx_class_highest_score]

    output_bbox = [min_x, min_y, max_x, max_y, class_highest_score, class_highest_score_name]
    return output_bbox

def get_YOLO_output(YOLO, Image, Classes_inDataSet):
    image_cv = cv2.cvtColor(Image, cv2.Color_RBG2BGR)
    image_cv_resized = cv2.resize(image_cv, (224, 224))

    image_cv_normalized = image_cv_resized / 255.0
    image_cv_normalized = np.expand_dims(image_cv_normalized, axis=0)
    image_cv_normalized = image_cv_normalized.astype('float32')

    YOLO_output = YOLO(image_cv_normalized)[0].numpy()

    cleaness = 0

    bbox_list = []
    for y in range(0, 7):
        for x in range(0, 7):
            bbox1_class_score = YOLO_output[y][x][10:] * YOLO_output[y][x][4]
            bbox2_class_score = YOLO_output[y][x][10:] * YOLO_output[y][x][9]

            bbox1 = YOLO_output[y][x][0:4]
            bbox2 = YOLO_output[y][x][5:9]

            process_bbox1 = process_bbox(x, y, bbox1, (224, 224), bbox1_class_score, Classes_inDataSet)
            process_bbox2 = process_bbox(x, y, bbox2, (224, 224), bbox2_class_score, Classes_inDataSet)

            bbox_list.extend([process_bbox1, process_bbox2])

    nms_bbox_list = nms(bbox_list)

    if 0 <= len(nms_bbox_list) < 2:
      cleaness = 1
    elif 2 <= len(nms_bbox_list) < 4:
      cleaness = 2
    elif 4 <= len(nms_bbox_list) < 6:
      cleaness = 3
    elif 6 <= len(nms_bbox_list):
      cleaness =4
    return cleaness


app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)


# 예측
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # 이미지를 메모리(램)에서 직접 처리
    contents = file.file.read()  # 동기 방식으로 파일 읽기
    img = Image.open(io.BytesIO(contents))  # PIL 라이브러리 사용

    if img.mode == 'RGBA':
        img = img.convert('RGB')

    clean_a = get_YOLO_ouput(YOLO, img, classes_yolo)

    final_predict = 0

    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    classes = ['1', '2', '3', '4']
    predicted_class = classes[class_idx]

    final_predict = str(round((2*int(predicted_class)/3) + (int(clean_a)/3)))

    return JSONResponse(content={"predicted_class": final_predict})

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)