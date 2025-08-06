import cv2
import os

def record_video(source=0, output_path='output.mp4', duration=5, fps=20):
    cap = cv2.VideoCapture(source)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = int(duration * fps)
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('录制中，按q退出', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

def extract_frames(video_path, video_name, time_interval_seconds=2, save_dir='frames'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 25

    frame_interval = int(fps * time_interval_seconds)
    if frame_interval < 1:
        frame_interval = 1

    frames = []
    frame_count = 0
    saved_count = 0
    video_basename = os.path.splitext(video_name)[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # 使用更独特的文件名以避免覆盖
            frame_path = os.path.join(save_dir, f'{video_basename}_frame_{saved_count}.jpg')
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            saved_count += 1
        frame_count += 1
    cap.release()
    return frames