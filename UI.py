import io, sys
import numpy as np
import pandas as pd
import torch
import torchaudio
import faiss
import streamlit as st
import soundfile as sf

REC_AVAILABLE = True
try:
    from audiorecorder import audiorecorder
except Exception:
    try:
        from AudioRecorder import audiorecorder
    except Exception:
        REC_AVAILABLE = False
        audiorecorder = None
#side bar
st.sidebar.write(f"Python: {sys.executable}")
import transformers
st.sidebar.write(f"Transformers: {transformers.__version__}")

try:
    from transformers import AutoProcessor, AutoModel
    PROC_CLS = AutoProcessor
    MODEL_CLS = AutoModel
except Exception:
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        PROC_CLS = Wav2Vec2Processor
        MODEL_CLS = Wav2Vec2Model
    except Exception:
        from transformers import AutoFeatureExtractor, Wav2Vec2Model
        PROC_CLS = AutoFeatureExtractor
        MODEL_CLS = Wav2Vec2Model

MODEL_ID = "facebook/wav2vec2-base-960h"
SR = 16000         # sampling rate
MAX_S = 2.0        # duration 2 seconds

def to_mono_16k(tensor: torch.Tensor, sr_in: int, max_s: float = MAX_S) -> torch.Tensor:
    """Make mono, resample to 16k, trim/pad to fixed length; returns shape [1, N]."""
    x = tensor
    # mono
    if x.ndim == 2 and x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    elif x.ndim == 1:
        x = x.unsqueeze(0)

    # resample
    if sr_in != SR:
        x = torchaudio.transforms.Resample(sr_in, SR)(x)

    # trim or pad
    n = int(max_s * SR)
    x = x[:, :n]
    if x.shape[1] < n:
        pad = torch.zeros((1, n - x.shape[1]), dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)

    # normalization
    peak = torch.max(torch.abs(x)).item()
    if peak > 0:
        x = 0.99 * x / peak
    return x

def metric_label_for(index_obj) -> str:
    if isinstance(index_obj, faiss.IndexFlatIP):
        return "cosine (IP)"
    if isinstance(index_obj, faiss.IndexFlatL2):
        return "L2 distance"
    return "score"

def load_audio_with_soundfile(file_bytes: bytes):
    #  Loading audio using soundfile.
    
    buf = io.BytesIO(file_bytes)
    audio_np, sr = sf.read(buf, dtype="float32", always_2d=True)  # [N, C]
    audio_np = audio_np.mean(axis=1)  # mono [N]
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)        # [1, N]
    return audio_tensor, sr

# Model & index  
@st.cache_resource
def load_model():
    processor = PROC_CLS.from_pretrained(MODEL_ID)
    model = MODEL_CLS.from_pretrained(MODEL_ID).eval()
    return processor, model

@st.cache_data
def load_index_and_metadata():
    try:
        index = faiss.read_index("wav2vec_faiss.index")
    except Exception:
        st.error("❌ FAISS index not found or invalid: 'wav2vec_faiss.index'. "
                 "Please run your indexing step to create it.")
        raise

    try:
        metadata = pd.read_csv("audio_metadata.csv")
    except Exception:
        st.error("❌ Missing 'audio_metadata.csv'. Please re-run preprocessing to generate it.")
        raise

    if hasattr(index, "ntotal") and index.ntotal != len(metadata):
        st.warning(f"⚠️ Index vectors ({index.ntotal}) != metadata rows ({len(metadata)}). "
                   "Make sure they were built from the same dataset/version.")
    return index, metadata

processor, model = load_model()
index, metadata = load_index_and_metadata()

# UI 
st.title("🎙 Audio Retrieval Demo")
st.caption("Upload or record a short digit (≈2s). The app embeds with Wav2Vec2 and searches the FAISS index.")

st.header("Upload or Record Your Query Audio")

col1, col2 = st.columns(2)
with col1:
    st.write("**Upload WAV file**")
    upload_file = st.file_uploader(" ", type=["wav"], label_visibility="collapsed")

with col2:
    st.write("**Or record here**")
    if REC_AVAILABLE:
        rec = audiorecorder("🎙 Start recording", "⏹ Stop recording")
    else:
        rec = None
        st.info("Recording is not available (audiorecorder module not found). You can still upload a WAV.")

audio_tensor, sr = None, None

#Upload path (soundfile)
if upload_file is not None:
    try:
        wav_bytes = upload_file.read()
        audio_tensor, sr = load_audio_with_soundfile(wav_bytes)
        st.audio(wav_bytes)
    except Exception as e:
        st.error(f"Could not read the uploaded WAV: {e}\n"
                 f"Tip: convert to PCM WAV (mono, 16kHz) and try again.")
        audio_tensor, sr = None, None

# Recording path (soundfile)
elif rec is not None and len(rec) > 0:
    try:
        # Export to bytes
        buf = io.BytesIO()
        rec = rec.set_channels(1).set_frame_rate(SR).set_sample_width(2) 
        rec.export(buf, format="wav")
        wav_bytes = buf.getvalue()

        st.audio(wav_bytes)

        audio_tensor, sr = load_audio_with_soundfile(wav_bytes)

    except Exception as e:
        st.error(f"Could not load the recorded audio: {e}")
        audio_tensor, sr = None, None

st.markdown("---")

# Retrieval
if audio_tensor is not None:
    st.write(f"Sample Rate (input): {sr} Hz")

    # preprocess to mono and fixed length
    audio_tensor = to_mono_16k(audio_tensor, sr, MAX_S)
    sr = SR

    # embedding
    with torch.inference_mode():
        inputs = processor(
            audio_tensor.squeeze().numpy(),
            sampling_rate=SR,
            return_tensors="pt"
        )
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype("float32")

    # normalizing for cosine 
    emb /= (np.linalg.norm(emb) + 1e-12)

    # searching
    k = st.slider("Top-K", 1, 20, 10)
    try:
        D, I = index.search(emb.reshape(1, -1), k)
    except Exception as e:
        st.error(f"Search failed: {e}")
        st.stop()

    # results
    metric = metric_label_for(index)
    st.subheader(f"Top {k} Retrieved Results")

    for rank, idx in enumerate(I[0], 1):
        ridx = int(idx)
        if ridx < 0 or ridx >= len(metadata):
            continue

        row = metadata.iloc[ridx]
        st.markdown(f"**Rank {rank}** — {metric}: `{float(D[0][rank-1]):.4f}`")
        st.write(f"Digit: {row.get('digit', 'NA')}")
        st.write(f"Accent: {row.get('accent', 'NA')}")
        st.write(f"Gender: {row.get('gender', 'NA')}")
        st.write(f"Age: {row.get('age', 'NA')}")
        st.write(f"Path: {row.get('path', 'NA')}")

        p = row.get("path")
        if isinstance(p, str) and len(p) > 0:
            try:
                st.audio(p)
            except Exception:
                st.info("Result audio preview unavailable (path not accessible).")

        st.markdown("---")
else:
    st.info("Upload a WAV or record audio to run retrieval.")



# # UI.py
# import io, sys
# import numpy as np
# import pandas as pd
# import torch, torchaudio
# import faiss
# import streamlit as st
# from AudioRecorder import audiorecorder


# # ================== Sidebar env info ==================
# st.sidebar.write(f"Python: {sys.executable}")
# import transformers
# st.sidebar.write(f"Transformers: {transformers.__version__}")

# # ================== HF imports (robust) ==================
# # Try modern AutoProcessor; fall back to explicit Wav2Vec2 classes; final fallback AutoFeatureExtractor
# try:
#     from transformers import AutoProcessor, AutoModel
#     PROC_CLS = AutoProcessor
#     MODEL_CLS = AutoModel
# except Exception:
#     try:
#         from transformers import Wav2Vec2Processor, Wav2Vec2Model
#         PROC_CLS = Wav2Vec2Processor
#         MODEL_CLS = Wav2Vec2Model
#     except Exception:
#         from transformers import AutoFeatureExtractor, Wav2Vec2Model
#         PROC_CLS = AutoFeatureExtractor      # works as processor for wav2vec2
#         MODEL_CLS = Wav2Vec2Model

# MODEL_ID = "facebook/wav2vec2-base-960h"
# SR = 16000         # single source of truth for sampling rate
# MAX_S = 2.0        # clamp query duration to 2s to keep inference snappy

# # ================== Helpers ==================
# def to_mono_16k(tensor: torch.Tensor, sr_in: int, max_s: float = MAX_S) -> torch.Tensor:
#     """Make mono, resample to 16k, trim/pad to fixed length; returns shape [1, N]."""
#     x = tensor
#     # mono
#     if x.ndim == 2 and x.shape[0] > 1:
#         x = x.mean(dim=0, keepdim=True)
#     elif x.ndim == 1:
#         x = x.unsqueeze(0)
#     # resample
#     if sr_in != SR:
#         x = torchaudio.transforms.Resample(sr_in, SR)(x)
#     # trim/pad
#     n = int(MAX_S * SR)
#     x = x[:, :n]
#     if x.shape[1] < n:
#         pad = torch.zeros((1, n - x.shape[1]), dtype=x.dtype)
#         x = torch.cat([x, pad], dim=1)
#     # peak normalize (safety)
#     peak = torch.max(torch.abs(x)).item()
#     if peak > 0:
#         x = 0.99 * x / peak
#     return x

# def metric_label_for(index_obj) -> str:
#     if isinstance(index_obj, faiss.IndexFlatIP):
#         return "cosine (IP)"
#     if isinstance(index_obj, faiss.IndexFlatL2):
#         return "L2 distance"
#     return "score"

# # ================== Model / index loaders ==================
# @st.cache_resource
# def load_model():
#     processor = PROC_CLS.from_pretrained(MODEL_ID)
#     model = MODEL_CLS.from_pretrained(MODEL_ID).eval()
#     return processor, model

# @st.cache_data
# def load_index_and_metadata():
#     try:
#         index = faiss.read_index("wav2vec_faiss.index")
#     except Exception as e:
#         st.error("❌ FAISS index not found or invalid: 'wav2vec_faiss.index'. "
#                  "Please run your indexing step to create it.")
#         raise
#     try:
#         metadata = pd.read_csv("audio_metadata.csv")
#     except Exception:
#         st.error("❌ Missing 'audio_metadata.csv'. Please re-run preprocessing to generate it.")
#         raise
#     # light check: index size vs metadata rows
#     if hasattr(index, "ntotal") and index.ntotal != len(metadata):
#         st.warning(f"⚠️ Index vectors ({index.ntotal}) != metadata rows ({len(metadata)}). "
#                    "Make sure they were built from the same dataset/version.")
#     return index, metadata

# processor, model = load_model()
# index, metadata = load_index_and_metadata()

# # ================== UI ==================
# st.title("🎙 Audio Retrieval Demo")
# st.caption("Upload or record a short digit (≈2s). The app embeds with Wav2Vec2 and searches the FAISS index.")

# st.header("Upload or Record Your Query Audio")
# col1, col2 = st.columns(2)
# with col1:
#     st.write("**Upload WAV file**")
#     upload_file = st.file_uploader(" ", type=["wav"], label_visibility="collapsed")
# with col2:
#     st.write("**Or record here**")
#     rec = audiorecorder("🎙 Start recording", "⏹ Stop recording")

# audio_tensor, sr = None, None

# # ---- Uploaded file path ----
# if upload_file is not None:
#     try:
#         audio_tensor, sr = torchaudio.load(upload_file)
#     except Exception:
#         st.error("Could not read the uploaded WAV. Ensure it's PCM WAV and try again.")
#         audio_tensor, sr = None, None
#     else:
#         st.audio(upload_file)

# # ---- Recording path ----
# elif rec is not None and len(rec) > 0:
#     # Convert pydub AudioSegment to a proper 16k mono 16-bit WAV in memory
#     buf = io.BytesIO()
#     rec = rec.set_channels(1).set_frame_rate(SR).set_sample_width(2)  # 16-bit
#     rec.export(buf, format="wav")
#     buf.seek(0)
#     st.audio(buf)  # preview the clean WAV (not raw bytes)
#     try:
#         audio_tensor, sr = torchaudio.load(buf)
#     except Exception:
#         st.error("Could not load the recorded audio. Please try recording again.")
#         audio_tensor, sr = None, None

# st.markdown("---")

# # ================== Retrieval ==================
# if audio_tensor is not None:
#     st.write(f"Sample Rate (input): {sr} Hz")

#     # preprocess
#     audio_tensor = to_mono_16k(audio_tensor, sr, MAX_S)
#     sr = SR

#     # embedding
#     with torch.inference_mode():
#         inputs = processor(
#             audio_tensor.squeeze().numpy(),
#             sampling_rate=SR,
#             return_tensors="pt"
#         )
#         outputs = model(**inputs)
#         # last_hidden_state: [B, T, H] -> mean pool over time -> [H]
#         emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype("float32")
#     # L2-normalize for cosine search
#     emb /= (np.linalg.norm(emb) + 1e-12)

#     # search
#     k = st.slider("Top-K", 1, 20, 10)
#     try:
#         D, I = index.search(emb.reshape(1, -1), k)
#     except Exception as e:
#         st.error(f"Search failed: {e}")
#         st.stop()

#     # results
#     metric = metric_label_for(index)
#     st.subheader(f"Top {k} Retrieved Results")
#     for rank, idx in enumerate(I[0], 1):
#         ridx = int(idx)
#         if ridx < 0 or ridx >= len(metadata):
#             continue
#         row = metadata.iloc[ridx]
#         st.markdown(f"**Rank {rank}** — {metric}: `{float(D[0][rank-1]):.4f}`")
#         st.write(f"Digit: {row.get('digit', 'NA')}")
#         st.write(f"Accent: {row.get('accent', 'NA')}")
#         st.write(f"Gender: {row.get('gender', 'NA')}")
#         st.write(f"Age: {row.get('age', 'NA')}")
#         st.write(f"Path: {row.get('path', 'NA')}")
#         # Streamlit can play local file paths if accessible
#         p = row.get("path")
#         if isinstance(p, str) and len(p) > 0:
#             try:
#                 st.audio(p)
#             except Exception:
#                 st.info("Result audio preview unavailable (path not accessible).")
#         st.markdown("---")
# else:
#     st.info("Upload a WAV or record audio to run retrieval.")
