import pathlib
from dorsal.testing import run_model
from dorsal_whisper.config import DORSAL_CONFIG 

TEST_ASSETS = pathlib.Path(__file__).parent / "assets"

def test_model_integration():
    """Tests the Whisper model running inside the Dorsal harness."""
    audio_file = TEST_ASSETS / "OSR_uk_000_0020_8k.wav"

    result = run_model(
        annotation_model=DORSAL_CONFIG["model_class"],
        file_path=str(audio_file),
        schema_id=DORSAL_CONFIG["schema_id"],
        validation_model=DORSAL_CONFIG.get("validation_model"),
        dependencies=DORSAL_CONFIG.get("dependencies"),
        options=DORSAL_CONFIG.get("options"),
    )

    assert result.error is None, f"Model execution failed: {result.error}"
    assert result.record is not None, "Model returned no data"

    output = result.record
    
    assert "faster-whisper" in output["producer"]
    assert "text" in output
    assert len(output["text"]) > 0, "Transcription should not be empty"
    assert len(output["segments"]) > 0
    assert "duration" in output
    assert output["duration"] > 0

    first_segment = output["segments"][0]
    assert "start_time" in first_segment
    assert "end_time" in first_segment