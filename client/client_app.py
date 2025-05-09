import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from io import BytesIO

# Configure page
st.set_page_config(page_title="AI Image Analyzer", layout="wide")
st.title("🧠 AI Image Analyzer (Client)")
st.write("Upload an image and choose tasks to analyze via FastAPI server")

# Server configuration
SERVER_URL = "http://localhost:30080/analyze"  # Update if hosted elsewhere

# Select tasks with descriptions
with st.expander("ℹ️ Task Descriptions"):
    st.markdown("""
    - **Detection**: Identifies objects with bounding boxes (YOLOv5)
    - **Classification**: Predicts ImageNet classes (ResNet50)
    - **Segmentation**: Pixel-level labeling (DeepLabV3)
    - **Features**: Extracts feature vectors (ResNet50)
    """)

tasks = st.multiselect(
    "Select analysis tasks",
    options=["detection", "classification", "segmentation", "features"],
    default=["detection"],
    help="Choose which analyses to perform"
)

# Image upload section
uploaded_file = st.file_uploader(
    "Upload an image (JPEG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Display original image
    st.subheader("Original Image")
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

# Analysis button
if st.button("🚀 Analyze Image", disabled=not uploaded_file):
    if not tasks:
        st.warning("Please select at least one task")
        st.stop()

    with st.spinner("Analyzing..."):
        try:
            # Prepare request
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            params = [("tasks", task) for task in tasks]
            
            # Send request
            response = requests.post(SERVER_URL, files=files, data=params)
            response.raise_for_status()
            results = response.json()["results"]

            st.success("Analysis Complete!")
            st.balloons()

            # Display results
            if "detection" in results:
                st.subheader("Object Detection Results")
                detection_img = np.array(image.copy())
                
                for obj in results["detection"]:
                    x1, y1, x2, y2 = map(int, obj["bbox"])
                    label = f"{obj['label']} ({obj['confidence']:.2f})"
                    
                    # Draw bounding box
                    cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(detection_img, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
                
                st.image(detection_img, caption="Detected Objects", use_column_width=True)
                st.json(results["detection"])

            if "classification" in results:
                st.subheader("Classification Results")
                cols = st.columns(2)
                with cols[0]:
                    st.table(sorted(
                        results["classification"],
                        key=lambda x: x["confidence"],
                        reverse=True
                    ))
                with cols[1]:
                    fig, ax = plt.subplots()
                    ax.barh(
                        [x["label"] for x in results["classification"]],
                        [x["confidence"] for x in results["classification"]]
                    )
                    ax.set_xlabel("Confidence")
                    st.pyplot(fig)

            if "segmentation" in results:
                st.subheader("Segmentation Results")
                if "mask" in results["segmentation"]:
                    mask = np.array(results["segmentation"]["mask"])
                    colored_mask = cv2.applyColorMap(
                        (mask * 10).astype(np.uint8), 
                        cv2.COLORMAP_JET
                    )
                    overlay = cv2.addWeighted(
                        np.array(image), 0.7, 
                        colored_mask, 0.3, 0
                    )
                    st.image(overlay, caption="Segmentation Overlay", use_column_width=True)
                else:
                    st.warning(results["segmentation"].get("error", "No segmentation results"))

            if "features" in results:
                st.subheader("Feature Vector")
                features = np.array(results["features"])
                
                # Top 20 most activated features
                top_k = min(20, len(features))
                idx = np.argsort(np.abs(features))[-top_k:][::-1]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(range(top_k), features[idx])
                ax.set_xticks(range(top_k))
                ax.set_xticklabels([f"F{i}" for i in idx], rotation=45)
                ax.set_title(f"Top {top_k} Feature Activations")
                st.pyplot(fig)
                
                # Download feature vector
                st.download_button(
                    label="📥 Download Features as CSV",
                    data=BytesIO(np.array2string(features, separator=',').encode()),
                    file_name="features.csv",
                    mime="text/csv"
                )

        except requests.exceptions.ConnectionError:
            st.error("🔌 Could not connect to server. Is it running?")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Server error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Sidebar with server status
with st.sidebar:
    st.header("Server Status")
    if st.button("Ping Server"):
        try:
            resp = requests.get(SERVER_URL.replace("/analyze", ""))
            if resp.status_code == 200:
                st.success("✅ Server is online!")
            else:
                st.warning(f"⚠️ Server responded with {resp.status_code}")
        except:
            st.error("❌ Server unavailable")
    
    st.markdown("---")
    st.markdown("**Model Information**")
    st.markdown("- YOLOv5 (Detection)")
    st.markdown("- ResNet50 (Classification/Features)")
    st.markdown("- DeepLabV3 (Segmentation)")