import pathlib
import tomllib
from dorsal.testing import run_model
from dorsal_whisper.model import FasterWhisperTranscriber

TEST_ASSETS = pathlib.Path(__file__).parent / "assets"

root = pathlib.Path(__file__).parent.parent
with open(root / "model_config.toml", "rb") as f:
    config = tomllib.load(f)


def test_model_integration():
    """Tests the Whisper model running inside the Dorsal harness."""
    audio_file = TEST_ASSETS / "OSR_uk_000_0020_8k.wav"

    result = run_model(
        annotation_model=FasterWhisperTranscriber,
        file_path=str(audio_file),
        schema_id=config["schema_id"],
        validation_model=config.get("validation_model"),
        dependencies=config.get("dependencies"),
        options=config.get("options"),
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
