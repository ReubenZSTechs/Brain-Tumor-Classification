import streamlit as st
from PIL import Image
from inference import predict

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image and the model will classify the tumor type.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Running inference..."):
            label, confidences = predict(uploaded_file)

        st.subheader("Prediction")
        st.success(label)

        st.subheader("Confidence Scores")
        for cls, score in confidences.items():
            st.write(f"{cls}: {score:.2f}%")

st.warning("⚠️ For educational purposes only. Not a medical diagnosis.")