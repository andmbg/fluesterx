"""Simple test for WhisperX transcription API."""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_URL = os.getenv("TEST_API_URL", "http://localhost:19000")
API_TOKEN = os.getenv("API_TOKEN")
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_transcription():
    """Test that transcription returns segments."""
    # Find audio file
    for file in FIXTURES_DIR.iterdir():
        if file.suffix.lower() in [".mp3", ".wav", ".m4a", ".flac"]:
            audio_file = file
            language = file.stem.split("_")[-1]
    
        with open(audio_file, "rb") as f:
            response = requests.post(
                f"{API_URL}/asr",
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                files={"audio_file": (audio_file.name, f)},
                data={"language": language},
                timeout=300
            )
        
        # Check response
        assert response.status_code == 200, f"Failed: {response.text}"
        
        result = response.json()
        assert "segments" in result
        assert len(result["segments"]) > 0
        assert result["segments"][0]["text"].strip()

        print(f"✅ Test passed! Got {len(result['segments'])} segments in language '{language}'.")


if __name__ == "__main__":
    test_transcription()
