from dorsal.file.dependencies import make_media_type_dependency
from .model import FasterWhisperTranscriber

DORSAL_CONFIG = {
    "model_class": FasterWhisperTranscriber,
    "schema_id": "open/audio-transcription",
    "dependencies": [make_media_type_dependency(include=["audio", "video"])],
    "options": {"model_size": "base"},
}
