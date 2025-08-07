import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import os
from config.logging_config import setup_logging
from models.model_loader import load_model
from models.model_tester import test_model_inference
from utils.image_processor import process_image
import torch
from ultralytics import YOLO
import onnxruntime as ort
import pandas as pd
from utils.video_frame_utils import record_video, extract_frames
import yaml
import hashlib

# ========== 工具函数 =========
def get_image_hash(image):
    """获取图片内容的哈希值，用于去重"""
    img_bytes = image.tobytes()
    return hashlib.md5(img_bytes).hexdigest()

def export_detection_records_to_csv():
    """导出检测记录到CSV文件，并保存带框的检测                    💡 **特别说明：**
                    - CSV文件中包含所有检测图片的绝对路径
                    - 图片文件名格式：检测时间_序号.jpg（如：20250807_145754_001.jpg）
                    - 可直接通过CSV文件中的路径打开对应图片"""
    if 'detection_records' not in st.session_state or not st.session_state.detection_records:
        return None
    
    # 生成时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 获取导出基础路径
    export_base_path = st.session_state.get('export_base_path', os.path.join(os.path.expanduser("~"), "Desktop", "corn_detection_exports"))
    
    # 创建基础导出文件夹（如果不存在）
    os.makedirs(export_base_path, exist_ok=True)
    
    # 创建此次导出的具体文件夹
    archive_folder = f"corn_detection_export_{timestamp}"
    archive_path = os.path.join(export_base_path, archive_folder)
    os.makedirs(archive_path, exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame(st.session_state.detection_records)
    
    # 添加检测结果图片绝对路径列
    df['检测结果图片绝对路径'] = ''
    
    # 统计保存成功的图片数量
    saved_images_count = 0
    failed_images_count = 0
    
    # 保存带框的检测结果图片
    if 'result_images_cache' in st.session_state:
        for idx, record in enumerate(st.session_state.detection_records):
            img_name = record['图片名称']
            detection_time = record['检测时间']
            
            # 使用检测时间作为文件名（替换特殊字符）
            safe_time = detection_time.replace(':', '').replace('-', '').replace(' ', '_')
            # 添加序号确保文件名唯一，统一使用.jpg扩展名
            result_filename = f"{safe_time}_{idx+1:03d}.jpg"
            result_save_path = os.path.join(archive_path, result_filename)
            
            # 保存检测结果图
            if img_name in st.session_state.result_images_cache:
                result_img_array = st.session_state.result_images_cache[img_name]
                try:
                    # 确保图片格式正确
                    if len(result_img_array.shape) == 3 and result_img_array.shape[2] == 3:
                        # BGR to RGB for saving
                        result_img_rgb = cv2.cvtColor(result_img_array, cv2.COLOR_BGR2RGB)
                        result_pil = Image.fromarray(result_img_rgb)
                    else:
                        result_pil = Image.fromarray(result_img_array)
                    
                    result_pil.save(result_save_path)
                    # 使用绝对路径
                    df.loc[idx, '检测结果图片绝对路径'] = os.path.abspath(result_save_path)
                    saved_images_count += 1
                    
                except Exception as e:
                    st.warning(f"保存检测结果图失败: {img_name}, 错误: {e}")
                    df.loc[idx, '检测结果图片绝对路径'] = f'保存失败: {str(e)}'
                    failed_images_count += 1
            else:
                df.loc[idx, '检测结果图片绝对路径'] = '图片缓存不存在'
                failed_images_count += 1
    
    # 保存CSV文件到同一文件夹
    csv_filename = f"corn_detection_records_{timestamp}.csv"
    csv_filepath = os.path.join(archive_path, csv_filename)
    df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
    
    # 创建简要说明文件
    readme_content = f"""玉米质量检测导出说明
===================

导出时间: {time.strftime("%Y年%m月%d日 %H:%M:%S")}
导出位置: {os.path.abspath(archive_path)}

统计信息:
- 检测图片总数: {len(st.session_state.detection_records)} 张
- 成功保存图片: {saved_images_count} 张
- 保存失败图片: {failed_images_count} 张

文件说明:
- {csv_filename}: 检测记录CSV文件（包含检测结果图片绝对路径）
- *.jpg: 带检测框的结果图片（以检测时间命名）

CSV文件字段说明:
- 图片名称: 原始图片文件名
- 检测时间: 检测执行的时间
- 检测结果: 检测到的坏粒和杂质详情
- 检测对象总数: 检测到的对象数量
- 置信度阈值: 使用的检测置信度阈值
- 检测结果图片绝对路径: 带检测框的图片文件的完整绝对路径

注意: 
1. 图片文件以检测时间命名（格式：YYYYMMDD_HHMMSS_序号.jpg），便于按时间排序和管理
2. CSV文件中的路径为绝对路径，可直接在文件管理器中打开
3. 所有检测到的图片都会被保存（不仅仅是预览的几张）
"""
    
    readme_path = os.path.join(archive_path, "导出说明.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    return csv_filepath, df, archive_path, saved_images_count, failed_images_count

def add_detection_record(image_name, detected_objects, detection_time, total_objects=0):
    """添加检测记录"""
    if 'detection_records' not in st.session_state:
        st.session_state.detection_records = []
    
    record = {
        '图片名称': image_name,
        '检测时间': detection_time,
        '检测结果': detected_objects if detected_objects else '未检测到坏粒或杂质',
        '检测对象总数': total_objects,
        '置信度阈值': st.session_state.get('current_confidence', 0.6)
    }
    
    st.session_state.detection_records.append(record)

# ========== 日志配置 =========
logger = setup_logging()

# ========== 页面与样式 =========
st.set_page_config(
    page_title="粮食质量检测平台",
    page_icon="🌽",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    :root {
        --primary-color: #4CAF50; /* Green */
        --secondary-color: #E8F5E9; /* Light Green */
        --accent-color: #FFC107; /* Amber */
        --background-color: #FFFFFF; /* White */
    }
    body { color: var(--primary-color); background-color: var(--background-color);}
    h1, h2, h3 { color: var(--primary-color);}
    .stButton > button {
        background-color: var(--accent-color);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stTextInput>label, .stNumberInput>label { color: var(--primary-color);}
    .streamlit-expanderHeader { color: var(--primary-color);}
</style>
""", unsafe_allow_html=True)

# ========== 标题与介绍 =========
st.title("🌽 粮食质量检测")
st.markdown("本平台基于深度学习技术，能够自动检测玉米中的坏粒和杂质，助力玉米品质管理和筛选。")

# ========== 侧边栏：模型设置 =========
with st.sidebar:
    st.header("检测模型设置")
    DEFAULT_MODEL_PATH = 'model/best.pt'
    default_model_exists = os.path.exists(DEFAULT_MODEL_PATH)

    if default_model_exists:
        st.info(f"检测到默认模型: {DEFAULT_MODEL_PATH}")
    else:
        st.warning(f"未找到默认模型: {DEFAULT_MODEL_PATH}")

    model_choice = st.radio("选择模型来源", ["默认模型", "上传自定义模型"])
    model_file = None
    model_type = None

    if model_choice == "默认模型" and default_model_exists:
        try:
            model_file = open(DEFAULT_MODEL_PATH, 'rb')
            file_ext = os.path.splitext(DEFAULT_MODEL_PATH)[1].lower()
            model_type = "ONNX" if file_ext == '.onnx' else "PyTorch"
            st.success("已选择默认模型")
        except Exception as e:
            st.error(f"无法加载默认模型: {e}")
            model_file = None
    elif model_choice == "上传自定义模型":
        model_file = st.file_uploader("上传模型文件", type=["pt", "pth", "onnx"])
        if model_file:
            file_ext = os.path.splitext(model_file.name)[1].lower()
            default_model_type = "ONNX" if file_ext == '.onnx' else "PyTorch"
            model_type = st.selectbox(
                "模型类型",
                ["PyTorch", "TorchScript", "ONNX"],
                index=["PyTorch", "TorchScript", "ONNX"].index(default_model_type)
            )
            st.success(f"已上传模型: {model_file.name}")
        else:
            st.info("请上传模型文件")
    else:
        st.info("请选择模型来源")

    # 只有选择模型后才显示高级设置
    if model_file and model_type:
        confidence_threshold = st.slider(
            "置信度阈值", min_value=0.0, max_value=1.0, value=0.6, step=0.05
        )
        # 保存当前置信度到session state
        st.session_state.current_confidence = confidence_threshold
        from utils.image_processor import LABELS
        with st.expander("高级设置"):
            draw_bbox = st.checkbox("显示边界框", value=True)
            draw_label = st.checkbox("显示类别名称", value=True)
            draw_confidence = st.checkbox("显示置信度", value=True)
            line_thickness = st.slider("边界框线条粗细", min_value=1, max_value=10, value=2)
            st.markdown("**各类别标记颜色**")
            if "detection_colors" not in st.session_state or len(st.session_state.detection_colors) != len(LABELS):
                st.session_state.detection_colors = ["#FF0000", "#00FF00", "#0000FF", "#FF00FF", "#FFFF00", "#00FFFF", "#FFA500", "#800080", "#008000", "#FF69B4"]  # 红、绿、蓝、紫、黄、青、橙、紫、深绿、粉
            for idx, label in enumerate(LABELS):
                st.session_state.detection_colors[idx] = st.color_picker(
                    f"{label} 标记颜色", st.session_state.detection_colors[idx], key=f"color_{idx}"
                )
            detection_colors = st.session_state.detection_colors

    # 导出设置
    st.header("导出设置")
    with st.expander("📂 导出路径配置"):
        # 默认导出路径
        default_export_path = os.path.join(os.path.expanduser("~"), "Desktop", "corn_detection_exports")
        
        # 自定义导出路径
        custom_export_path = st.text_input(
            "自定义导出路径（可选）",
            value="",
            placeholder=f"留空使用默认路径: {default_export_path}",
            help="输入自定义的导出文件夹路径，留空则使用桌面上的默认文件夹"
        )
        
        # 保存导出路径到session state
        if custom_export_path.strip():
            st.session_state.export_base_path = custom_export_path.strip()
        else:
            st.session_state.export_base_path = default_export_path
        
        st.info(f"当前导出路径: {st.session_state.export_base_path}")
    
    st.header("关于")
    st.info("""
    本平台专为玉米坏粒和杂质检测设计，支持多种图片格式。
    上传玉米图片后，系统将自动检测并标记出各类坏粒和杂质。
    """)

# ========== 日志相关函数 =========
def log_event(event):
    """记录操作步骤和时间"""
    if 'log_data' not in st.session_state:
        st.session_state.log_data = {
            '序号': [],
            '操作步骤': [],
            '执行时间': []
        }
    st.session_state.log_data['序号'].append(len(st.session_state.log_data['序号']) + 1)
    st.session_state.log_data['操作步骤'].append(event)
    st.session_state.log_data['执行时间'].append(time.strftime("%Y-%m-%d %H:%M:%S"))

def show_log_table():
    """显示操作日志表格"""
    if 'log_data' in st.session_state and st.session_state.log_data['序号']:
        st.markdown("""
        <style>
        .corn-table-header {
            background: linear-gradient(45deg, #a8e063 0%, #56ab2f 100%); /* Green gradient */
            color: #ffffff;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        [data-testid="stDataFrame"] > div {
            background: #e8f5e9; /* Light green */
            padding: 1rem;
            border-radius: 10px;
            box_shadow: 0 2px 8px rgba(76, 175, 80, 0.15); /* Green shadow */
        }
        [data-testid="stDataFrame"] div[data-testid="stHorizontalBlock"] {
            background: white;
            margin: 0.2rem 0;
            border-radius: 5px;
            transition: transform 0.2s ease;
        }
        [data-testid="stDataFrame"] div[data-testid="stHorizontalBlock"]:hover {
            transform: translateX(5px);
            background: #f0fdf0; /* Lighter green on hover */
        }
        [data-testid="stDataFrame"] div[data-testid="stHorizontalBlock"] > div:last-child {
            font-family: monospace;
            color: #388E3C; /* Darker green */
        }
        [data-testid="stDataFrame"] div[data-testid="stHorizontalBlock"] > div:first-child {
            font-weight: 600;
            color: #1B5E20; /* Even darker green */
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="corn-table-header">🌽操作步骤记录</div>', unsafe_allow_html=True)
        df = pd.DataFrame(st.session_state.log_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "序号": st.column_config.NumberColumn(
                    "序号", help="操作步骤序号", format="%d", width="small"
                ),
                "操作步骤": st.column_config.TextColumn(
                    "操作步骤", help="执行的具体操作", width="medium"
                ),
                "执行时间": st.column_config.TextColumn(
                    "执行时间", help="操作执行的具体时间", width="medium"
                )
            }
        )

def show_category_table(label_counts=None):
    """显示美食类别统计表"""
    with open("data.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    categories = data["names"]
    cn_categories = data.get("cn_names", categories)
    df_data = []
    for i, cat in enumerate(categories):
        cn_cat = cn_categories[i] if i < len(cn_categories) else cat
        quantity = 0
        if label_counts and cn_cat in label_counts:
            quantity = label_counts[cn_cat]
        df_data.append([cn_cat, quantity])

    df = pd.DataFrame(df_data, columns=["类别", "数量"])

    st.markdown("### 玉米坏粒和杂质统计表")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "类别": st.column_config.TextColumn("类别", width="medium"),
            "数量": st.column_config.NumberColumn("数量", format="%d", width="small"),
        }
    )

# ========== Session State 初始化 =========
if 'log_data' not in st.session_state:
    st.session_state.log_data = {'序号': [], '操作步骤': [], '执行时间': []}
if 'latest_label_counts' not in st.session_state:
    st.session_state.latest_label_counts = None
if 'detection_records' not in st.session_state:
    st.session_state.detection_records = []
if 'result_images_cache' not in st.session_state:
    st.session_state.result_images_cache = {}
if 'export_base_path' not in st.session_state:
    st.session_state.export_base_path = os.path.join(os.path.expanduser("~"), "Desktop", "corn_detection_exports")

# ========== 模型加载 =========
if 'loaded_model' not in st.session_state or 'loaded_model_type' not in st.session_state:
    if model_file and model_type:
        log_event(f"加载模型: {model_type}")
        st.session_state.loaded_model = load_model(model_file, model_type)
        st.session_state.loaded_model_type = model_type
        if st.session_state.loaded_model and test_model_inference(st.session_state.loaded_model, model_type):
            log_event("模型推理测试通过")
        else:
            log_event("模型推理测试未通过")
    else:
        st.session_state.loaded_model = None
        st.session_state.loaded_model_type = None

# ========== 主界面布局 =========
col1, col2 = st.columns([1, 1])

with col1:
    num_per_row = 4
    status_message = st.empty()

    st.subheader("1. 上传文件")
    uploaded_media = st.file_uploader(
        "选择图片(jpg, png)或视频(mp4, mov, avi)",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "mov", "avi"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_media:
        for medium in uploaded_media:
            file_ext = os.path.splitext(medium.name)[1].lower()
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image = Image.open(medium)
                img_hash = get_image_hash(image)
                if "image_hashes" not in st.session_state:
                    st.session_state.image_hashes = set()
                if img_hash not in st.session_state.image_hashes:
                    img_path = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time())}_{medium.name}")
                    image.save(img_path)
                    if "frame_paths" not in st.session_state:
                        st.session_state.frame_paths = []
                    st.session_state.frame_paths.append(img_path)
                    st.session_state.image_hashes.add(img_hash)
                    log_event(f"上传图片成功: {medium.name}")
                else:
                    log_event(f"图片 {medium.name} 已存在，跳过")
            elif file_ext in ['.mp4', '.mov', '.avi']:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                tfile.write(medium.read())
                video_path = tfile.name
                log_event(f"上传视频成功: {medium.name}")
                with st.spinner(f"正在从 {medium.name} 抽帧..."):
                    frame_paths = extract_frames(video_path, medium.name, time_interval_seconds=2, save_dir='frames')
                if "frame_paths" not in st.session_state:
                    st.session_state.frame_paths = []
                st.session_state.frame_paths.extend([p for p in frame_paths if p not in st.session_state.frame_paths])
                log_event(f"从 {medium.name} 抽帧完成，共抽取 {len(frame_paths)} 张图片")
        status_message.success("文件处理完成，请在下方查看预览并开始识别。")

    st.subheader("2. 使用摄像头")
    use_camera = st.checkbox("开启摄像头")
    if use_camera:
        camera_img = st.camera_input("点击拍照")
        if camera_img:
            image = Image.open(camera_img)
            img_hash = get_image_hash(image)
            if "image_hashes" not in st.session_state:
                st.session_state.image_hashes = set()
            if img_hash not in st.session_state.image_hashes:
                img_path = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time())}_camera.png")
                image.save(img_path)
                if "frame_paths" not in st.session_state:
                    st.session_state.frame_paths = []
                st.session_state.frame_paths.append(img_path)
                st.session_state.image_hashes.add(img_hash)
                log_event(f"摄像头拍摄成功")
                status_message.success("照片已添加，请在下方查看预览并开始识别。")
            else:
                status_message.info("该照片已存在，无需重复添加。")

    st.subheader("3. 其他方式")
    remote_video_url = st.text_input('输入远程视频流URL (RTSP, RTMP, HTTP)')
    if st.button('处理5秒远程视频'):
        if remote_video_url:
            log_event(f"开始处理远程视频流: {remote_video_url}")
            with st.spinner('正在录制和抽帧...'):
                video_path = record_video(source=remote_video_url, output_path='remote_video.mp4', duration=5)
            log_event("远程视频录制完成，开始抽帧")
            st.video(video_path)
            with st.spinner("正在抽帧..."):
                frame_paths = extract_frames(video_path, 'remote_video', time_interval_seconds=2, save_dir='frames')
            if "frame_paths" not in st.session_state:
                st.session_state.frame_paths = []
            st.session_state.frame_paths.extend([p for p in frame_paths if p not in st.session_state.frame_paths])
            status_message.success(f"抽取 {len(frame_paths)} 帧，等待识别...")
            log_event(f"抽帧完成，共抽取 {len(frame_paths)} 张图片")
        else:
            st.warning('请输入有效的视频流URL')

    if st.button("录制5秒视频", key="record_video_btn"):
        log_event("开始录制视频")
        with st.spinner("正在录制视频..."):
            video_path = record_video(output_path='output.mp4', duration=5, fps=20)
        log_event("视频录制完成，开始抽帧")
        st.video(video_path)
        with st.spinner("正在抽帧..."):
            frame_paths = extract_frames(video_path, 'recorded_video', time_interval_seconds=2, save_dir='frames')
        if "frame_paths" not in st.session_state:
            st.session_state.frame_paths = []
        st.session_state.frame_paths.extend([p for p in frame_paths if p not in st.session_state.frame_paths])
        status_message.success(f"抽取 {len(frame_paths)} 帧，等待识别...")
        log_event(f"抽帧完成，共抽取 {len(frame_paths)} 张图片")

    if st.session_state.get("frame_paths"):
        st.session_state.frame_paths = list(dict.fromkeys(st.session_state.frame_paths))
        st.markdown("--- ")
        st.markdown("#### 待识别图片预览")
        all_preview = st.session_state.frame_paths

        def show_preview_grid(path_list):
            rows = (len(path_list) + num_per_row - 1) // num_per_row
            for row in range(rows):
                cols = st.columns(num_per_row)
                for col in range(num_per_row):
                    idx = row * num_per_row + col
                    if idx < len(path_list):
                        frame_path = path_list[idx]
                        try:
                            img = Image.open(frame_path)
                            with cols[col]:
                                st.image(img, caption=os.path.basename(frame_path), use_container_width=True)
                        except Exception as e:
                            with cols[col]:
                                st.warning(f"无法预览图片: {frame_path}，错误信息: {e}")

        preview_paths = all_preview[:num_per_row]
        show_preview_grid(preview_paths)
        more_paths = all_preview[num_per_row:]
        if more_paths:
            with st.expander(f"查看更多（共 {len(more_paths)} 张）"):
                show_preview_grid(more_paths)

        if st.button("开始识别", key="start_detect_btn"):
            status_message.info("正在识别中...")
            from collections import Counter
            from utils.image_processor import get_label
            total_label_counts = Counter()
            result_images = []
            start_time = time.time()
            log_event(f"开始识别，共 {len(st.session_state.frame_paths)} 张图片")
            
            # 清空之前的检测记录和缓存
            st.session_state.detection_records = []
            st.session_state.result_images_cache = {}

            for frame_path in st.session_state.frame_paths:
                image = Image.open(frame_path)
                img_array = np.array(image)
                img_name = os.path.basename(frame_path)
                detection_time = time.strftime("%Y-%m-%d %H:%M:%S")
                
                log_event(f"处理图片: {img_name}")
                if img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                    log_event("图片通道由RGBA转为RGB")
                else:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    log_event("图片通道由RGB转为BGR")
                    
                if st.session_state.loaded_model and st.session_state.loaded_model_type:
                    try:
                        result_img, filtered_class_ids = process_image(
                            img_array, st.session_state.loaded_model, st.session_state.loaded_model_type,
                            confidence_threshold, detection_colors,
                            draw_bbox, draw_label, draw_confidence, line_thickness
                        )
                        result_images.append((result_img, img_name))
                        
                        # 缓存检测结果图片
                        st.session_state.result_images_cache[img_name] = result_img.copy()
                        
                        if len(filtered_class_ids) > 0:
                            label_names = [get_label(cls_id) for cls_id in filtered_class_ids]
                            label_counts = Counter(label_names)
                            total_label_counts += label_counts
                            
                            # 格式化检测结果字符串
                            detected_objects = ", ".join([f"{label}({count}个)" for label, count in label_counts.items()])
                            total_objects = len(filtered_class_ids)
                            
                            log_event(f"{img_name} 识别成功: {dict(label_counts)}")
                            
                            # 添加检测记录
                            add_detection_record(img_name, detected_objects, detection_time, total_objects)
                        else:
                            log_event(f"{img_name} 未检测到坏粒或杂质")
                            # 添加空记录
                            add_detection_record(img_name, "未检测到坏粒或杂质", detection_time, 0)
                    except Exception as e:
                        log_event(f"{img_name} 处理失败: {str(e)}")
                        # 添加失败记录
                        add_detection_record(img_name, f"处理失败: {str(e)}", detection_time, 0)
                        # 使用原图作为结果，也要缓存
                        result_images.append((img_array, img_name))
                        st.session_state.result_images_cache[img_name] = img_array.copy()

            end_time = time.time()
            process_time = end_time - start_time
            st.session_state.latest_label_counts = total_label_counts
            log_event(f"识别统计完成: {dict(total_label_counts)}，处理 {len(st.session_state.frame_paths)} 张图片，共耗时 {process_time:.2f} 秒")
            status_message.success("识别完成！")

            st.markdown("#### 识别结果预览")
            all_results = result_images

            def show_image_grid(image_list, caption_prefix="识别结果"):
                rows = (len(image_list) + num_per_row - 1) // num_per_row
                for row in range(rows):
                    cols = st.columns(num_per_row)
                    for col in range(num_per_row):
                        idx = row * num_per_row + col
                        if idx < len(image_list):
                            img_arr, fname = image_list[idx]
                            with cols[col]:
                                # 确保图片格式正确
                                if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
                                    # BGR to RGB
                                    display_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                                else:
                                    display_img = img_arr
                                st.image(
                                    display_img,
                                    caption=f"{caption_prefix}: {fname}",
                                    use_container_width=True
                                )

            preview_results = all_results[:num_per_row]
            show_image_grid(preview_results)
            more_results = all_results[num_per_row:]
            if more_results:
                with st.expander(f"查看更多识别结果（共 {len(more_results)} 张）"):
                    show_image_grid(more_results)

with col2:
    st.subheader("操作日志")
    if 'log_data' in st.session_state:
        show_log_table()
    else:
        st.info("暂无操作日志记录")

# ========== 底部统计表 =========
st.markdown("---")
show_category_table(st.session_state.latest_label_counts)

# ========== 检测记录管理 =========
st.markdown("---")
st.markdown("### 📋 检测记录管理")

col_record1, col_record2 = st.columns([3, 1])

with col_record1:
    if 'detection_records' in st.session_state and st.session_state.detection_records:
        st.markdown("#### 当前检测记录")
        
        # 显示检测记录表格
        df_records = pd.DataFrame(st.session_state.detection_records)
        st.dataframe(
            df_records,
            use_container_width=True,
            hide_index=True,
            column_config={
                "图片名称": st.column_config.TextColumn("图片名称", width="medium"),
                "检测时间": st.column_config.TextColumn("检测时间", width="medium"),
                "检测结果": st.column_config.TextColumn("检测结果", width="large"),
                "检测对象总数": st.column_config.NumberColumn("检测对象总数", format="%d", width="small"),
                "置信度阈值": st.column_config.NumberColumn("置信度阈值", format="%.2f", width="small")
            }
        )
        
        st.markdown("💡 **说明：** 导出时，带检测框的图片将以检测时间加序号命名（如：20250807_145754_001.jpg），CSV文件将包含每张图片的绝对路径信息。")
        
        # 显示统计信息
        total_images = len(st.session_state.detection_records)
        images_with_objects = len([r for r in st.session_state.detection_records if r['检测对象总数'] > 0])
        total_objects = sum([r['检测对象总数'] for r in st.session_state.detection_records])
        cached_images = len(st.session_state.get('result_images_cache', {}))
        
        st.markdown(f"""
        **📊 本次检测统计：**
        - 总检测图片数：{total_images} 张
        - 有检测对象的图片：{images_with_objects} 张
        - 检测对象总数：{total_objects} 个
        - 检测准确率：{(images_with_objects/total_images*100):.1f}%
        - 缓存的结果图片：{cached_images} 张
        """)
        
    else:
        st.info("暂无检测记录。请先进行图片检测。")

with col_record2:
    st.markdown("#### 导出功能")
    
    # 显示当前导出路径设置
    current_export_path = st.session_state.get('export_base_path', os.path.join(os.path.expanduser("~"), "Desktop", "corn_detection_exports"))
    st.markdown(f"📁 **当前导出路径：** `{current_export_path}`")
    st.markdown("💡 *可在左侧边栏的'导出设置'中修改导出路径*")
    
    if 'detection_records' in st.session_state and st.session_state.detection_records:
        # 预览导出信息
        total_records = len(st.session_state.detection_records)
        cached_images = len(st.session_state.get('result_images_cache', {}))
        
        st.markdown("#### 📋 导出预览")
        st.markdown(f"""
        **即将导出：**
        - 📄 检测记录：{total_records} 条
        - 🖼️ 带框图片：{cached_images} 张
        - 💾 CSV文件：1 个（含绝对路径）
        - 📝 说明文件：1 个
        """)
        
        if cached_images < total_records:
            st.warning(f"⚠️ 注意：有 {total_records - cached_images} 张图片未缓存，可能无法导出对应的带框图片")
        
        st.markdown("---")
        
        # 生成预览的导出文件夹名
        preview_timestamp = time.strftime("%Y%m%d_%H%M%S")
        preview_folder_name = f"corn_detection_export_{preview_timestamp}"
        preview_full_path = os.path.join(current_export_path, preview_folder_name)
        
        st.markdown("#### 🚀 开始导出")
        st.markdown(f"**导出目标：** `{preview_full_path}`")
        
        if st.button("📤 一键导出", type="primary"):
            try:
                result = export_detection_records_to_csv()
                if result:
                    csv_filepath, df, archive_path, saved_images_count, failed_images_count = result
                    log_event(f"检测记录一键导出成功: {os.path.basename(archive_path)}")
                    st.success(f"✅ 导出成功！\n📁 导出文件夹：{archive_path}")
                    
                    # 显示详细的导出统计信息
                    st.info(f"""
                    📊 **导出内容详情：**
                    - 🖼️ 成功保存图片：{saved_images_count} 张
                    - ❌ 保存失败图片：{failed_images_count} 张
                    - 📄 检测记录CSV文件：1 个
                    - 📝 说明文件：1 个
                    - 📂 导出位置：{archive_path}
                    
                    � **特别说明：**
                    - CSV文件中包含所有检测图片的绝对路径
                    - 图片文件名格式：检测时间_原图名.扩展名
                    - 可直接通过CSV文件中的路径打开对应图片
                    """)
                    
                    # 如果有失败的图片，显示警告
                    if failed_images_count > 0:
                        st.warning(f"⚠️ 有 {failed_images_count} 张图片保存失败，请检查导出文件夹权限或磁盘空间。")
                    
                    # 提供下载CSV文件的按钮
                    with open(csv_filepath, 'rb') as f:
                        st.download_button(
                            label="🔽 下载CSV记录文件",
                            data=f.read(),
                            file_name=os.path.basename(csv_filepath),
                            mime='text/csv'
                        )
                    
                    # 提供文件夹位置信息和操作建议
                    st.markdown(f"💡 **完整导出位置：** `{archive_path}`")
                    st.markdown("🔍 您可以在文件管理器中打开此文件夹查看所有带检测框的图片。")
                    st.markdown("📋 CSV文件中的'检测结果图片绝对路径'列包含每张图片的完整路径。")
                    
                else:
                    st.error("❌ 导出失败！没有检测记录可导出。")
            except Exception as e:
                st.error(f"❌ 导出时出现错误：{str(e)}")
                logger.error(f"一键导出时出错: {e}")
    else:
        st.info("需要先进行检测才能导出记录")
    
    st.markdown("---")
    st.markdown("#### 🗑️ 清空数据")
    
    if st.button("🗑️ 清空所有记录", type="secondary"):
        # 统计要清空的数据
        records_count = len(st.session_state.get('detection_records', []))
        cache_count = len(st.session_state.get('result_images_cache', {}))
        
        st.session_state.detection_records = []
        st.session_state.result_images_cache = {}
        log_event(f"清空检测记录({records_count}条)和图片缓存({cache_count}张)")
        st.success(f"✅ 已清空 {records_count} 条检测记录和 {cache_count} 张缓存图片！")
        st.rerun()

