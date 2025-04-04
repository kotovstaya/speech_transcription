import uuid
import streamlit as st
from dotenv import load_dotenv
from frontend_utils import get_transcription

load_dotenv()


def set_button_clicked():
    st.session_state.button_clicked = True


if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


if "output_text" not in st.session_state:
    st.session_state.output_text = ""


if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False


st.title("Speech Transcription App")
st.divider()

target_dB = -20
target_peak = 0.95

uploaded_file = st.file_uploader("upload WAV file", type=["wav"])
chunk_size_sec = st.slider("Chunk size (sec)", 0, 30, value=30, step=1)
norm_algo = st.selectbox("Normalization algorithm:", ["as is", "peak", "rms", "lufs"])
if norm_algo == "peak":
    target_peak = st.slider('target peak:', 0.0, 1.0, value=0.95, step=0.01, format="%.2f")
if norm_algo in ["rms", "lufs"]:
    target_dB = st.slider("dB", -40, -10, value=-20, step=1)


text_output = st.text_area(
    label="Transcription",
    value=st.session_state.output_text,
    height=550,
    max_chars=None
)

st.button("Transcribe", on_click=set_button_clicked)

if uploaded_file is not None and st.session_state.button_clicked:
    st.session_state.output_text = ""
    st.session_state.button_clicked = False
    with st.spinner('Request in progress...'):
        st.session_state.output_text = get_transcription(
            uploaded_file=uploaded_file,
            user_id=st.session_state.user_id,
            chunk_size_sec=chunk_size_sec,
            target_dB=target_dB,
            norm_algo=norm_algo,
            target_peak=target_peak,
        )
        st.rerun()
