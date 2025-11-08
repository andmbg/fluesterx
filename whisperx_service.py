"""
Custom WhisperX service with support for WAV2VEC2_ASR_LARGE_LV60K_960H alignment model
"""

import os
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from dotenv import load_dotenv, find_dotenv

from src.whisperx import WhisperXService
from src.logger import logger

# Load environment variables from .env file
load_dotenv(find_dotenv())

logger.info(f"HF_TOKEN: {os.getenv('HF_TOKEN')}")

app = Flask(__name__)


# Global service instance
logger.info(f"DEVICE: {os.getenv('DEVICE', '(none)')}")
service = WhisperXService(device=os.getenv("DEVICE", "cpu")) # type: ignore


@app.route("/asr", methods=["POST"])
def transcribe():
    """Transcription endpoint"""
    try:
        # --- API Key check ---
        api_key = request.headers.get("Authorization")
        expected_key = f"Bearer {os.getenv('API_TOKEN')}"
        if expected_key and api_key != expected_key:
            return jsonify({"error": "Invalid or missing API key"}), 401
        # ----------------------

        # Check if audio file is provided
        if "audio_file" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            return jsonify({"error": "No audio file selected"}), 400

        # Get parameters from both form data and query params
        params = {}

        # Form data parameters
        for key in ["task", "language", "model"]:
            if key in request.form:
                params[key] = request.form[key]

        # Query parameters (these take precedence)
        for key in ["diarize", "min_speakers", "max_speakers", "output", "model"]:
            if key in request.args:
                params[key] = request.args[key]

        logger.info(f"Transcription request with parameters: {params}")

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(audio_file.filename).suffix
        ) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Perform transcription
            result = service.transcribe(tmp_path, params)

            # Clean up
            os.unlink(tmp_path)

            return jsonify(result)

        except Exception:
            # Clean up on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    except Exception as e:
        logger.error(f"Transcription request processing error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "message": "Custom WhisperX API with WAV2VEC2_ASR_LARGE_LV60K_960H alignment",
            "device": service.device,
            "models_loaded": {
                "whisper": service.model is not None,
                "alignment": service.align_model is not None,
                "diarization": service.diarize_model is not None,
            },
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 19000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"HF_TOKEN: {os.getenv('HF_TOKEN')}")
    logger.info(f"Starting WhisperX service on {host}:{port}")
    app.run(host=host, port=port, debug=True, threaded=True)
