import io
import os

import librosa
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_transcription(chunk_size_sec: int, uploaded_file) -> None:
    st.session_state.output_text = ""
    st.session_state.button_clicked = False
    model_sr = 16000
    with st.spinner('Request in progress...'):
        file_bytes = io.BytesIO(uploaded_file.read())
        audio, original_sr = librosa.load(file_bytes, sr=None)
        audio = librosa.resample(y=audio, orig_sr=original_sr, target_sr=model_sr)  # noqa: E501
        chunk_length = chunk_size_sec * model_sr
        for i, chunk in enumerate([audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]):  # noqa: E501
            resp = requests.post(
                os.getenv("BACKEND_ENDPOINT"),
                json={"chunk": chunk.tolist()},
                headers={"Content-Type": "application/json"},
            )
            st.session_state.output_text += resp.json()["response"]


def set_button_clicked():
    st.session_state.button_clicked = True


if "output_text" not in st.session_state:
    st.session_state.output_text = ""


if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False


st.title("Speech Transcription App")
st.divider()


uploaded_file = st.file_uploader("upload WAV file", type=["wav"])
chunk_size_sec = st.slider("Chunk size (sec)", 0, 30, value=30, step=1)


text_output = st.text_area(
    label="Transcription",
    value=st.session_state.output_text,
    height=550,
    max_chars=None
)

st.button("Transcribe", on_click=set_button_clicked)

if uploaded_file is not None and st.session_state.button_clicked:
    get_transcription(chunk_size_sec, uploaded_file)
    st.rerun()
