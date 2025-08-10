import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from deep_translator import GoogleTranslator

# Load BLIP model
@st.cache_resource
def load_model():
    st.write("⏳ Loading model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Using device: {device}")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

st.title("🧭 Landmark Identifier with Translation")
st.markdown("Upload a photo of a landmark and get a cultural/historic caption in your chosen language.")

# Language selection
lang_dict = {
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml"
}
lang_choice = st.selectbox("Choose translation language", list(lang_dict.keys()))
lang_code = lang_dict[lang_choice]

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Identifying landmark..."):
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Translation using deep-translator
        try:
            translated_caption = GoogleTranslator(source='auto', target=lang_code).translate(caption)
        except Exception as e:
            translated_caption = f"Translation failed: {e}"

        st.success("Landmark Identified!")
        st.markdown(f"**English Description:** {caption}")
        st.markdown(f"**{lang_choice} Translation:** {translated_caption}")
