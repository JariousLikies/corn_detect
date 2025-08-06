import logging
import numpy as np
import cv2
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from utils.example_generator import generate_example_result
from utils.color_converter import hex_to_rgb

logger = logging.getLogger(__name__)

# ============ 标签加载 =============
def load_labels(yaml_path='data.yaml'):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data['cn_names']
    except Exception as e:
        logger.warning(f"无法加载标签文件: {e}")
        return [f"类别{i}" for i in range(10)]

LABELS = load_labels()

def get_label(class_id):
    try:
        return LABELS[int(class_id)]
    except Exception:
        return f"类别{class_id}"

# ============ 图像预处理 =============
def preprocess_image(image):
    # 假设模型输入为640x640
    target_size = 640
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh))
    new_image = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    new_image[:nh, :nw] = image_resized
    img_tensor = torch.from_numpy(new_image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return img_tensor, (h, w, nh, nw, scale)

# ============ 主推理函数 =============
def process_image(image, model, model_type, threshold, color_hex_list, draw_bbox, draw_label, draw_confidence, line_thickness):
    import streamlit as st
    if model is None:
        st.warning("使用示例检测结果，因为没有加载有效模型")
        # 只传一个颜色字符串
        return generate_example_result(
            image, threshold, color_hex_list[0] if isinstance(color_hex_list, list) else color_hex_list,
            draw_bbox, draw_label, draw_confidence, line_thickness
        )

    try:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor, original_info = preprocess_image(img_rgb)

        with torch.no_grad():
            if model_type == "ONNX":
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: img_tensor.numpy()})
                st.write("模型原始输出：", outputs)
                # 你需要根据你的ONNX模型输出格式自行实现后处理
                # 这里只给出空结果
                boxes = np.array([])
                scores = np.array([])
                class_ids = np.array([])
            else:
                outputs = model(img_tensor)
                results = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

                boxes_obj = results.boxes
                if boxes_obj is not None and boxes_obj.shape[0] > 0:
                    boxes = boxes_obj.xyxy.cpu().numpy()
                    scores = boxes_obj.conf.cpu().numpy()
                    class_ids = boxes_obj.cls.cpu().numpy().astype(int)
                else:
                    # 初始化为空的二维数组以匹配维度
                    boxes = np.empty((0, 4))
                    scores = np.array([])
                    class_ids = np.array([])

            # 1. 创建置信度掩码
            conf_mask = scores >= threshold

            # 2. 创建类别ID掩码 (只保留ID为0和1的类别)
            class_mask = (class_ids == 0) | (class_ids == 1)

            # 3. 结合两个掩码
            final_mask = conf_mask & class_mask

            # 应用最终掩码，布尔索引能保持维度正确性
            filtered_boxes = boxes[final_mask]
            filtered_scores = scores[final_mask]
            filtered_class_ids = class_ids[final_mask]

            logger.info(f"模型原始输出类别ID: {np.unique(class_ids)}")
            logger.info(f"过滤后类别ID: {filtered_class_ids}")

        img_with_boxes = img_rgb.copy()
        h, w, nh, nw, scale = original_info

        # 坐标转换：模型输出的坐标是基于640x640的预处理图像，
        # 预处理时将原图等比缩放后放置在左上角，因此只需按比例缩放回去即可。
        filtered_boxes[:, [0, 2]] /= scale
        filtered_boxes[:, [1, 3]] /= scale

        # 裁剪到图像边界
        filtered_boxes[:, [0, 2]] = np.clip(filtered_boxes[:, [0, 2]], 0, w)
        filtered_boxes[:, [1, 3]] = np.clip(filtered_boxes[:, [1, 3]], 0, h)

        # 使用PIL绘制中文
        pil_img = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("simhei.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        for box, score, class_id in zip(filtered_boxes, filtered_scores, filtered_class_ids):
            x1, y1, x2, y2 = map(int, box)
            if x1 >= x2 or y1 >= y2:
                continue

            class_id_int = int(class_id)
            if class_id_int >= len(LABELS):
                logger.warning(f"未知的类别ID: {class_id_int}, 将使用默认标签和颜色。")
                label = f"未知类别{class_id_int}"
                color_hex = "#FF0000" # 红色表示未知类别
            else:
                label = get_label(class_id_int)
                color_hex = color_hex_list[class_id_int % len(color_hex_list)] # 使用取模防止索引越界

            color = hex_to_rgb(color_hex)
            color = hex_to_rgb(color_hex)

            if draw_bbox:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_thickness)

            if draw_label or draw_confidence:
                text = label
                if draw_confidence:
                    text += f": {score:.2f}"

                text_bbox = draw.textbbox((x1, y1), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                text_y = y1 - text_h - 5 if y1 - text_h - 5 > 0 else y1 + 5
                draw.rectangle([x1, text_y, x1 + text_w, text_y + text_h], fill=color)
                draw.text((x1, text_y), text, fill="white", font=font)

        img_with_boxes = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        print("推理后类别ID：", filtered_class_ids)
        return img_with_boxes, filtered_class_ids

    except Exception as e:
        import streamlit as st
        st.error(f"图像处理失败: {e}")
        logger.error(f"图像处理错误: {e}", exc_info=True)
        # 只传一个颜色字符串
        return generate_example_result(
            image, threshold, color_hex_list[0] if isinstance(color_hex_list, list) else color_hex_list,
            draw_bbox, draw_label, draw_confidence, line_thickness
        )
