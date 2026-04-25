import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import json
from pipeline import VehicleIntelligencePipeline

# Ensure models load only once
@st.cache_resource
def load_pipeline():
    return VehicleIntelligencePipeline()

def main():
    st.set_page_config(page_title="KnightSight ANPR Dashboard", layout="wide")
    st.title("🚗 Edge-Optimized Vehicle Intelligence")
    st.markdown("Upload an image to test the end-to-end pipeline: Vehicle Detection -> Plate Localization -> ANPR.")

    st.sidebar.header("Settings")
    st.sidebar.markdown("This dashboard showcases the lightweight inference pipeline optimized for edge deployments.")

    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            is_video = file_extension in ['mp4', 'avi', 'mov']
            
            st.markdown("### Processing...")
            
            with st.spinner("Running Inference Pipeline..."):
                pipeline = load_pipeline()
                
                if is_video:
                    import tempfile
                    import os
                    
                    # Save uploaded video to temp file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    
                    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    output_path = output_file.name
                    output_file.close()  # Close it so OpenCV can write to it on Windows
                    
                    st.info("Video inference active. Enhancing frames, detecting vehicles, tracking plates...")
                    progress_bar = st.progress(0)
                    
                    def update_progress(pct):
                        progress_bar.progress(pct)
                    
                    # Process video
                    results = pipeline.process_video(tfile.name, output_path, skip_frames=2, progress_callback=update_progress)
                    
                    progress_bar.empty()
                    st.success("Video processing complete!")
                    
                    st.markdown("### Annotated Video")
                    
                    # Read the video bytes to display in streamlit
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        if len(video_bytes) > 0:
                            st.video(video_bytes)
                        else:
                            st.error("Error: Video processing failed. The output video is empty. This could be a codec issue with OpenCV 'mp4v'.")
                    
                    # Cleanup input temp file
                    try:
                        os.unlink(tfile.name)
                        os.unlink(output_path)
                    except:
                        pass
                else:
                    # Convert uploaded file to OpenCV format
                    image = Image.open(uploaded_file)
                    image = ImageOps.exif_transpose(image).convert('RGB')
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    results, vehicles, plates = pipeline.process_image(image_array=img_bgr)
                    annotated_img = pipeline.annotate_image(img_bgr, results, vehicles)
                    
                    # Convert back to RGB for Streamlit
                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(annotated_rgb, caption="Annotated Result", use_container_width=True)

            st.markdown("### Structured Output")
            st.json(results)

        except Exception as e:
            import traceback
            st.error(f"Error processing file: {e}")
            st.text(traceback.format_exc())

if __name__ == '__main__':
    main()
