import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
from PIL import Image
import os

st.set_page_config(page_title="No-Ball Detector", layout="wide")
st.title("🏏 AI-Powered No-Ball Detection Using Pose Estimation")
st.markdown("This app detects **illegal elbow bending** during fast bowling using pose estimation and ICC rules (max 15° elbow extension).")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    video_file = st.file_uploader("📂 Upload Bowling Video", type=["mp4", "mov", "avi"])
    selected_arm = st.selectbox("🎯 Bowling Arm", ["Right", "Left"])
    threshold = st.slider("⚠️ Max Legal Elbow Angle (°)", 145, 180, 165)
    show_each_frame = st.checkbox("📸 Show All Frames", value=True)

# Helper: Calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return round(angle, 2)

# Pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

if video_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(video_file.read())

    cap = cv2.VideoCapture(temp_file.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    stframe = st.empty()
    progress = st.progress(0)
    stats = {"total": 0, "no_balls": 0}
    no_ball_images = []

    with mp_pose.Pose(static_image_mode=False) as pose:
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            stats["total"] += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark

                # Choose arm based on user input
                if selected_arm == "Right":
                    shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                             lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                else:
                    shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Convert to pixels
                shoulder = np.multiply(shoulder, [frame_width, frame_height]).astype(int)
                elbow = np.multiply(elbow, [frame_width, frame_height]).astype(int)
                wrist = np.multiply(wrist, [frame_width, frame_height]).astype(int)

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle < threshold:
                    label = f"❌ No-Ball (Elbow: {angle}°)"
                    stats["no_balls"] += 1
                    color = (0, 0, 255)
                    # Save this frame
                    if frame_count % 5 == 0:
                        no_ball_images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                else:
                    label = f"✅ Legal Ball (Elbow: {angle}°)"
                    color = (0, 255, 0)

                cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Show current frame
            if show_each_frame:
                stframe.image(frame, channels="BGR", caption=f"Frame {frame_count}")
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()

    # 📊 Summary
    st.subheader("📈 Analysis Summary")
    total = stats["total"]
    no_balls = stats["no_balls"]
    legal = total - no_balls
    percent = (no_balls / total) * 100 if total > 0 else 0

    st.write(f"🔢 Total Frames: {total}")
    st.write(f"✅ Legal Deliveries: {legal}")
    st.write(f"❌ No-Ball Frames: {no_balls} ({percent:.2f}%)")

    # 🖼️ Display saved no-ball images
    if no_ball_images:
        st.subheader("🛑 No-Ball Frames Detected")
        st.image(no_ball_images, caption=[f"No-Ball Frame {i+1}" for i in range(len(no_ball_images))], width=300)

    st.success("✅ Video analysis completed!")
