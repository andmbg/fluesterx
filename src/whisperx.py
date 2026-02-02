import os
import traceback
from typing import Literal

import torch
import whisperx

# PyTorch 2.6+ defaults weights_only=True in torch.load for security.
# Pyannote model checkpoints use many custom classes (omegaconf, typing, etc.)
# that aren't in the safe globals list. Since we trust HuggingFace models,
# we patch torch.load to default to weights_only=False.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # Force disable, lightning_fabric passes True explicitly
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from src.logger import logger


class WhisperXService:
    def __init__(self, device: Literal["cpu", "cuda", "auto"] = "auto", default_language: str = "en"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.default_language = default_language

        logger.info(f"Initializing WhisperX service on {self.device}")
        logger.info(f"Default language: {self.default_language}")

        # Load default model
        self.model = None
        self.align_model_cache = {}  # Cache alignment models per language
        self.diarize_model = None

        self._load_models()

    def _load_models(self):
        """Load WhisperX models"""
        try:
            # Load default Whisper model
            model_name = os.getenv("ASR_MODEL", "base")
            logger.info(f"Loading Whisper model: {model_name}")
            self.model = whisperx.load_model(
                model_name, self.device, compute_type=self.compute_type
            )

            # Don't load alignment model here - load it per-language on demand
            logger.info("Alignment models will be loaded on demand per language")

            # Load diarization model with better error handling
            hf_token = os.getenv("HF_TOKEN")
            logger.info(f"HF_TOKEN available: {hf_token is not None}")

            if hf_token:
                logger.info("Attempting to load diarization model...")
                try:
                    logger.info(
                        f"WhisperX available functions: {[attr for attr in dir(whisperx) if 'diar' in attr.lower()]}"
                    )

                    if hasattr(whisperx, "DiarizationPipeline"):
                        logger.info("Using WhisperX DiarizationPipeline")
                        self.diarize_model = whisperx.DiarizationPipeline(
                            use_auth_token=hf_token, device=self.device
                        )
                    elif hasattr(whisperx, "load_diarization_model"):
                        logger.info("Using whisperx.load_diarization_model")
                        self.diarize_model = whisperx.load_diarization_model(
                            use_auth_token=hf_token, device=self.device
                        )
                    else:
                        logger.info("Using pyannote.audio directly")
                        from pyannote.audio import Pipeline

                        self.diarize_model = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1", token=hf_token
                        ).to(torch.device(self.device))

                    logger.info("Diarization model loaded successfully")

                except Exception as e:
                    logger.error(f"Failed to load diarization model: {e}")
                    logger.error(traceback.format_exc())
                    self.diarize_model = None
            else:
                logger.warning("No HF_TOKEN provided, diarization will be disabled")
                self.diarize_model = None

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(traceback.format_exc())
            raise

    def _get_alignment_model(self, language_code: str):
        """Load and cache alignment model for a specific language"""
        if language_code not in self.align_model_cache:
            logger.info(f"Loading alignment model for language: {language_code}")
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device,
                )
                self.align_model_cache[language_code] = (align_model, align_metadata)
                logger.info(f"Alignment model loaded for {language_code}")
            except Exception as e:
                logger.error(f"Failed to load alignment model for {language_code}: {e}")
                raise
        
        return self.align_model_cache[language_code]

    def transcribe(self, audio_file_path, params):
        """Perform transcription with alignment and optional diarization"""
        try:
            # Load audio
            audio = whisperx.load_audio(audio_file_path)

            model_name = os.getenv("ASR_MODEL", "base")
            if hasattr(self, "_current_model") and self._current_model != model_name:
                logger.info(f"Switching model to {model_name}")
                self.model = whisperx.load_model(
                    model_name, self.device, compute_type=self.compute_type
                )
                self._current_model = model_name
            elif not hasattr(self, "_current_model"):
                self._current_model = model_name

            # Get language parameter (use default if not specified)
            language = params.get("language", self.default_language)
            logger.info(f"Transcribing with language: {language}")

            # Transcribe
            logger.info("Starting transcription")
            result = self.model.transcribe(audio, batch_size=16, language=language)

            # Get detected or specified language
            detected_language = result.get("language", language)
            logger.info(f"Detected language: {detected_language}")

            # Load appropriate alignment model for the language
            align_model, align_metadata = self._get_alignment_model(detected_language)

            # Align with language-specific model
            logger.info(f"Starting alignment for language: {detected_language}")
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            # Diarization if requested and available
            diarize = params.get("diarize", "false").lower() == "true"
            if diarize and self.diarize_model:
                logger.info("Starting diarization")
                min_speakers = int(params.get("min_speakers", 2))
                max_speakers = int(params.get("max_speakers", 5))

                try:
                    logger.info("Running diarization with pyannote pipeline")
                    diarize_segments = self.diarize_model(
                        {
                            "waveform": torch.from_numpy(audio[None, :]).float(),
                            "sample_rate": 16000,
                        },
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )

                    diarize_df = self._convert_pyannote_to_whisperx(diarize_segments)
                    result = whisperx.assign_word_speakers(diarize_df, result)

                except Exception as e:
                    logger.warning(f"Diarization failed: {e}")
                    logger.warning(traceback.format_exc())
                    result["diarization_error"] = str(e)

            elif diarize and not self.diarize_model:
                logger.warning(
                    "Diarization requested but no diarization model available"
                )

            # Add metadata
            result["language"] = detected_language
            result["model"] = model_name
            result["alignment_language"] = detected_language
            result["diarization_enabled"] = diarize and self.diarize_model is not None

            # Filter out unnecessary information
            result = self._clean_result(result, params)

            logger.info("Transcription completed successfully")
            return result

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            logger.error(traceback.format_exc())
            raise

    def _convert_pyannote_to_whisperx(self, diarization):
        """Convert pyannote diarization output to WhisperX format"""
        import pandas as pd

        diarization_list = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            diarization_list.append(
                {"start": segment.start, "end": segment.end, "speaker": speaker}
            )

        return pd.DataFrame(diarization_list)

    def _clean_result(self, result, params):
        """Remove unnecessary information from the result"""

        # Check if user wants minimal output
        minimal = params.get("minimal", "false").lower() == "true"
        include_scores = params.get("include_scores", "false").lower() == "true"
        include_word_timestamps = (
            params.get("include_word_timestamps", "true").lower() == "true"
        )

        if "segments" in result:
            cleaned_segments = []

            for segment in result["segments"]:
                cleaned_segment = {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text", "").strip(),
                }

                # Add speaker if available (diarization)
                if "speaker" in segment:
                    cleaned_segment["speaker"] = segment["speaker"]

                # Optionally include scores
                if include_scores and "score" in segment:
                    cleaned_segment["score"] = segment["score"]

                # Optionally include word-level timestamps
                if include_word_timestamps and "words" in segment:
                    cleaned_words = []
                    for word in segment["words"]:
                        cleaned_word = {
                            "word": word.get("word", ""),
                            "start": word.get("start"),
                            "end": word.get("end"),
                        }

                        # Add speaker for word-level diarization
                        if "speaker" in word:
                            cleaned_word["speaker"] = word["speaker"]

                        # Optionally include word scores
                        if include_scores and "score" in word:
                            cleaned_word["score"] = word["score"]

                        cleaned_words.append(cleaned_word)

                    cleaned_segment["words"] = cleaned_words

                cleaned_segments.append(cleaned_segment)

            result["segments"] = cleaned_segments

        # Keep only essential metadata
        if minimal:
            essential_result = {
                "segments": result.get("segments", []),
                "language": result.get("language"),
                "text": result.get("text", ""),  # Full text if available
            }
            return essential_result

        # Remove technical metadata but keep useful info
        result.pop("word_segments", None)
        result.pop("char_segments", None)

        return result
