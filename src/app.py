import numpy as np
import streamlit as st
from inference import presentation_inf

st.title("Handwriting Synthesis Demo")
text = st.text_input("Input Text")

if st.button("Generate!"):
    image = presentation_inf(text)
    image = (image + 1) / 2
    st.image(np.transpose(image, axes=(1, 2, 0)))
