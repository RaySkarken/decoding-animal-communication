"""Feature extraction: LogMelSpectrogram layer + preprocess_model builder."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from tqdm.auto import tqdm

from .data import resample_audio, pad_mean


class LogMelSpectrogram(keras.layers.Layer):
    """Compute log-magnitude mel-scaled spectrograms (verbatim from Assom / TF_AE.ipynb)."""

    def __init__(
        self,
        sample_rate: int,
        fft_size: int,
        hop_size: int,
        fft_length: int,
        window_fn,
        n_mels: int,
        f_min: float = 0.0,
        f_max: float | None = None,
        normalize: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.fft_length = fft_length
        self.window_fn = window_fn
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.normalize = normalize
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.fft_length // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max,
        )

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super().build(input_shape)

    def call(self, waveforms):
        def _tf_log10(x):
            num = tf.math.log(x)
            den = tf.math.log(tf.constant(10, dtype=num.dtype))
            return num / den

        def _normalize(log_mel, scaler=self.normalize):
            min_v = tf.math.reduce_min(log_mel, axis=3, keepdims=True)
            max_v = tf.math.reduce_max(log_mel, axis=3, keepdims=True)
            if scaler == "tanh":
                out = 2.0 * (log_mel - min_v) / (max_v - min_v + 1e-7) - 1.0
                nan_val = -1.0
            elif scaler == "sigmoid":
                out = (log_mel - min_v) / (max_v - min_v + 1e-7)
                nan_val = 0.0
            else:
                return log_mel
            idx = tf.where(tf.math.is_nan(out))
            out = tf.tensor_scatter_nd_update(
                out, idx, tf.ones(tf.shape(idx)[0], dtype=out.dtype) * nan_val,
            )
            return out

        def power_to_db(magnitude, amin=1e-16, top_db=120.0):
            ref = tf.reduce_max(magnitude)
            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
            return log_spec

        spectrograms = tf.signal.stft(
            waveforms,
            frame_length=self.fft_size,
            frame_step=self.hop_size,
            fft_length=self.fft_length,
            pad_end=True,
        )
        magnitude = tf.abs(spectrograms)
        mel = tf.matmul(tf.square(magnitude), self.mel_filterbank)
        log_mel = power_to_db(mel)
        log_mel = _normalize(log_mel)
        sh = tf.shape(log_mel)
        return tf.reshape(log_mel, [-1, sh[2], sh[3]])

    def get_config(self):
        config = {
            "fft_size": self.fft_size,
            "hop_size": self.hop_size,
            "fft_length": self.fft_length,
            "window_fn": self.window_fn,
            "n_mels": self.n_mels,
            "sample_rate": self.sample_rate,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "normalize": self.normalize,
        }
        config.update(super().get_config())
        return config


def build_preprocess_model(
    max_len: int,
    sample_rate: int = 250_000,
    fft_size: int = 1024,
    hop_size: int = 1024,
    fft_length: int = 16384,
    n_mels: int = 128,
    f_min: float = 500,
    f_max: float = 120_000,
    normalize: str = "tanh",
    norm_adapt_data: np.ndarray | None = None,
) -> keras.Model:
    """Build the TF preprocessing model matching TF_AE.ipynb.

    Returns a compiled Keras model: waveform (max_len,) -> (time, n_mels).
    If *norm_adapt_data* is provided, the Normalization layer is adapted on it.
    """
    inp = keras.Input(shape=(max_len,), dtype=tf.float32, name="waveform")
    x = keras.layers.Reshape((1, -1), name="flatten_to_1d")(inp)

    log_mel_layer = LogMelSpectrogram(
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_size=hop_size,
        fft_length=fft_length,
        window_fn=tf.signal.hamming_window,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        normalize=normalize,
        name="LogMel",
    )
    x = log_mel_layer(x)

    norm_layer = keras.layers.Normalization(axis=-1, name="Normalization")
    x = norm_layer(x)

    model = keras.Model(inp, x, name="preprocess_model")
    model.compile()

    if norm_adapt_data is not None:
        mel_out = log_mel_layer(
            keras.layers.Reshape((1, -1))(
                tf.constant(norm_adapt_data, dtype=tf.float32)
            )
        )
        norm_layer.adapt(mel_out.numpy())

    return model


def compute_spectrograms(
    seg_df,
    target_sr: int = 250_000,
    spec_time: int = 21,
    spec_freq: int = 32,
    batch_size: int = 32,
    norm_adapt_n: int = 2048,
    **model_kwargs,
) -> np.ndarray:
    """End-to-end: seg_df with 'audio'/'sr' columns -> (N, spec_time, spec_freq) array."""
    max_len = 0
    for idx in range(len(seg_df)):
        row = seg_df.iloc[idx]
        y = resample_audio(row["audio"], int(row["sr"]), target_sr)
        max_len = max(max_len, len(y))

    padded = []
    for idx in range(len(seg_df)):
        row = seg_df.iloc[idx]
        y = resample_audio(row["audio"], int(row["sr"]), target_sr)
        padded.append(pad_mean(y, max_len))
    padded = np.stack(padded, axis=0).astype(np.float32)

    adapt_n = min(norm_adapt_n, len(padded))
    model = build_preprocess_model(
        max_len=max_len,
        sample_rate=target_sr,
        norm_adapt_data=padded[:adapt_n],
        **model_kwargs,
    )

    n = len(padded)
    mel_all = []
    for start in tqdm(range(0, n, batch_size), desc="TF preprocess_model"):
        batch = padded[start : start + batch_size]
        out = model.predict(batch, verbose=0)
        mel_all.append(out)
    mel_all = np.concatenate(mel_all, axis=0)

    tf_specs = np.zeros((n, spec_time, spec_freq), dtype=np.float32)
    for i in range(n):
        tf_specs[i] = resize(mel_all[i], (spec_time, spec_freq), anti_aliasing=True)

    return tf_specs


# ---------------------------------------------------------------------------
# SSL Embeddings (optional -- requires torch + transformers)
# ---------------------------------------------------------------------------

def compute_ssl_embeddings(
    seg_df,
    model_name: str = "facebook/hubert-base-ls960",
    target_sr: int = 16_000,
    batch_size: int = 8,
    device: str | None = None,
) -> np.ndarray:
    """Extract SSL embeddings (mean-pooled) from a HuggingFace audio model.

    Supports HuBERT, Wav2Vec 2.0, or any AutoModel that returns ``last_hidden_state``.
    Segments are resampled to *target_sr* (typically 16 kHz for speech SSL models).

    Returns (N, hidden_dim) float32 array.
    """
    import torch
    from transformers import AutoModel, AutoFeatureExtractor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings = []
    n = len(seg_df)

    for start in tqdm(range(0, n, batch_size), desc=f"SSL ({model_name})"):
        batch_audio = []
        for idx in range(start, min(start + batch_size, n)):
            row = seg_df.iloc[idx]
            y = resample_audio(row["audio"], int(row["sr"]), target_sr)
            batch_audio.append(y)

        inputs = extractor(
            batch_audio, sampling_rate=target_sr, return_tensors="pt",
            padding=True, truncation=True, max_length=target_sr * 30,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1).cpu().numpy()
        embeddings.append(pooled)

    return np.concatenate(embeddings, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# BEATs Embeddings (requires torch + torchaudio + src.beats)
# ---------------------------------------------------------------------------

def ensure_naturelm_beats_merged(
    microsoft_checkpoint_path: str,
    naturelm_dir: str,
) -> dict:
    """Merge NatureLM-audio ``beats.*`` weights into a full BEATs ``state_dict``.

    The Hugging Face ``model.safetensors`` matches Microsoft BEATs except it omits
    ``encoder.layers.{1..11}.self_attn.relative_attention_bias.weight`` (only
    layer 0 is stored). Those tensors are filled from *microsoft_checkpoint_path*.
    The 527-way AS2M predictor head is ignored; ``finetuned_model`` is forced
    False so :meth:`BEATs.extract_features` returns 768-D patch states.

    Writes ``beats_encoder_merged.pt`` under *naturelm_dir* on first call and
    reloads it thereafter.
    """
    import json
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    nl = Path(naturelm_dir)
    merged_path = nl / "beats_encoder_merged.pt"
    if merged_path.is_file():
        return torch.load(merged_path, map_location="cpu", weights_only=False)

    cfg_path = nl / "config.json"
    st_path = nl / "model.safetensors"
    if not cfg_path.is_file() or not st_path.is_file():
        raise FileNotFoundError(
            f"NatureLM-audio folder must contain config.json and model.safetensors: {nl}"
        )

    cfg_all = json.loads(cfg_path.read_text(encoding="utf-8"))
    beats_cfg: dict = dict(cfg_all["beats_cfg"])
    beats_cfg["finetuned_model"] = False

    ms_ckpt = torch.load(microsoft_checkpoint_path, map_location="cpu", weights_only=False)
    merged: dict = dict(ms_ckpt["model"])
    nat_flat = {
        k[6:]: v
        for k, v in load_file(str(st_path)).items()
        if k.startswith("beats.")
    }
    for key, tensor in nat_flat.items():
        if key in merged and key not in ("predictor.weight", "predictor.bias"):
            merged[key] = tensor.clone()

    blob = {"model": merged, "cfg": beats_cfg}
    torch.save(blob, merged_path)
    return blob


def compute_beats_embeddings(
    seg_df,
    checkpoint_path: str,
    mode: str = "naive",
    native_sr: int = 250_000,
    target_sr: int = 16_000,
    batch_size: int = 8,
    device: str | None = None,
    *,
    encoder: str = "microsoft",
    naturelm_dir: str | None = None,
) -> np.ndarray:
    """Extract BEATs embeddings (mean-pooled) from pretrained checkpoint.

    Parameters
    ----------
    seg_df : DataFrame with 'audio' (np.ndarray) and 'sr' (int) columns.
    checkpoint_path : path to ``BEATs_iter3_plus_AS2M.pt`` (Microsoft AS2M base).
        Required for ``encoder="microsoft"``. For ``encoder="naturelm"`` it is
        still required so missing ``relative_attention_bias`` slices can be copied.
    mode : ``"naive"`` — resample to *target_sr* (loses ultrasonic content);
           ``"pitch_shift"`` — slow the audio so the full 0–Nyquist range
           maps into 0–(*target_sr*/2), preserving spectral content.
    native_sr : original sample rate of the recordings (250 kHz for fruit bats).
    target_sr : sample rate expected by BEATs (16 kHz).
    batch_size : segments per forward pass.
    device : ``"cpu"`` or ``"cuda"``; auto-detected if None.
    encoder : ``"microsoft"`` — original AS2M checkpoint only; ``"naturelm"`` —
        EarthSpeciesProject/NatureLM-audio encoder (merged with Microsoft fill).
    naturelm_dir : directory containing ``config.json`` + ``model.safetensors``
        from NatureLM-audio; required when *encoder* is ``"naturelm"``.

    Returns
    -------
    ndarray of shape (N, 768), float32.
    """
    import torch
    from .beats.BEATs import BEATs, BEATsConfig

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if encoder == "naturelm":
        if not naturelm_dir:
            raise ValueError('naturelm_dir is required when encoder="naturelm"')
        blob = ensure_naturelm_beats_merged(checkpoint_path, naturelm_dir)
        cfg = BEATsConfig(blob["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(blob["model"], strict=True)
    elif encoder == "microsoft":
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(ckpt["model"])
    else:
        raise ValueError(f'encoder must be "microsoft" or "naturelm", got {encoder!r}')

    model.to(device).eval()

    slowdown = native_sr // target_sr  # e.g. 250000 // 16000 = 15

    # BEATs patch embedding is 16x16; fbank uses 25ms frames / 10ms shift at
    # 16 kHz, so we need >= 16 frames -> 25 + 15*10 = 175 ms -> 2800 samples.
    MIN_SAMPLES_16K = 2800

    embeddings = []
    n = len(seg_df)

    for start in tqdm(range(0, n, batch_size), desc=f"BEATs ({encoder}/{mode})"):
        batch_audio = []
        max_len = 0

        for idx in range(start, min(start + batch_size, n)):
            row = seg_df.iloc[idx]
            y = np.asarray(row["audio"], dtype=np.float32)
            sr = int(row["sr"])

            if mode == "pitch_shift":
                y = resample_audio(y, sr, sr // slowdown)
            else:
                y = resample_audio(y, sr, target_sr)

            if len(y) < MIN_SAMPLES_16K:
                y = np.pad(y, (0, MIN_SAMPLES_16K - len(y)))

            batch_audio.append(y)
            max_len = max(max_len, len(y))

        padded = np.zeros((len(batch_audio), max_len), dtype=np.float32)
        padding_mask = np.ones((len(batch_audio), max_len), dtype=bool)
        for i, y in enumerate(batch_audio):
            padded[i, : len(y)] = y
            padding_mask[i, : len(y)] = False

        wav_tensor = torch.from_numpy(padded).to(device)
        mask_tensor = torch.from_numpy(padding_mask).to(device)

        with torch.no_grad():
            features, _ = model.extract_features(wav_tensor, padding_mask=mask_tensor)

        pooled = features.mean(dim=1).cpu().numpy()
        embeddings.append(pooled)

    return np.concatenate(embeddings, axis=0).astype(np.float32)
