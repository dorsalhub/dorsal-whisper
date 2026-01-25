# Dorsal Whisper

A [Dorsal](https://dorsalhub.com) model wrapper for **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)**.

This plugin allows you to integrate high-performance speech-to-text transcription directly into your Dorsal processing pipelines using CTranslate2-based Whisper models.

## Features

* **Fast Inference:** Uses `faster-whisper` (CTranslate2) for up to 4x speed improvements over the original OpenAI implementation.
* **Automatic VAD:** Built-in Voice Activity Detection to filter silence and improve accuracy.
* **Schema Compliant:** Outputs standardized JSON matching the `open/audio-transcription` schema.
* **Safe Caching:** Efficient model loading that prevents memory leaks during batch processing.

## Compatibility

* **Python:** 3.11, 3.12, 3.13
* **Note:** Python 3.14 is **not supported** due to upstream dependencies (`onnxruntime` and `faster-whisper`) that are not yet compatible with 3.14.

## Installation

```bash
pip install dorsal-whisper

```

*Note: You may need to install NVIDIA libraries (cuBLAS/cuDNN) separately if you intend to run on GPU. See the [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper) for GPU setup.*


### Default Configuration

Uses the default `base` model on `device="auto"`.

```toml
[models]
  [models.whisper]
  id = "github:dorsalhub/dorsal-whisper"

```

## Output

This model produces a file annotation conforming to the [Open Validation Schemas](http://github.com/dorsalhub/open-validation-schemas) Audio Transcription schema:

* **Schema ID:** `open/audio-transcription` (v0.3.0)
* **Key Fields:**
* `text`: The full concatenated transcript.
* `segments`: An array of timed segments with `start_time`, `end_time`, and `score`.
* `language`: The detected ISO-639-3 language code (e.g., `eng`).
* `duration`: The total duration of the source media in seconds.
* `attributes`: Includes `language_probability`.



## Development

### Running Tests

This repository uses `pytest` for integration testing.

```bash
pip install -e .[test]
pytest

```

## License

This project is licensed under the Apache 2.0 License.

```
