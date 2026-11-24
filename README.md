# speech2text

On-device English speech recognition in Rust: microphone capture with
[cpal](https://github.com/RustAudio/cpal) and OpenAI's Whisper running on
[candle](https://github.com/huggingface/candle). It uses the Metal GPU on macOS,
an NVIDIA GPU when built with the `cuda` feature, and the CPU otherwise. The Whisper encoder-decoder, the log-mel front end
(matching Hugging Face's `WhisperFeatureExtractor`) and the resampler are
written directly in this crate.

```sh
cargo run --release --example transcribe -- tests/data/buy_milk.wav
cargo run --release --example transcribe -- --record 5         # from the default microphone
cargo run --release --example transcribe -- --checkpoint small --cpu file.wav
```

```
audio: 3.46 s, loudest frame 0.2794 RMS
loaded large-v3-turbo on Metal(MetalDevice(DeviceId(1))) in 2.09 s
warmed up in 8.82 s
transcribed in 0.59 s (5.9x real time), confidence 0.962, no-speech 0.000
Remind me to buy milk tomorrow morning, it is important.
```

The numbers above are from an M2 Max. The first Metal warm-up compiles the
kernels, so it's slow once and fast after that. The same goes for CUDA, whose
driver caches the compiled kernels between runs: on an RTX 5080 the first
warm-up takes 15 s, later ones 0.3 s, and the clip transcribes in 0.11 s.

## Checkpoints

| name             | Hugging Face repo               | download |
|------------------|---------------------------------|----------|
| `large-v3-turbo` | `openai/whisper-large-v3-turbo` | 1.6 GB   |
| `small`          | `openai/whisper-small`          | 1.0 GB   |

`large-v3-turbo` is the default. Each checkpoint is pinned to a fixed revision,
so the weights stay the ones the code was tested with. They download into the
Hugging Face cache, which is found the same way `huggingface_hub` finds it:
`$HF_HUB_CACHE`, `$HUGGINGFACE_HUB_CACHE`, `$HF_HOME/hub`,
`$XDG_CACHE_HOME/huggingface/hub`, then `~/.cache/huggingface/hub`.
`$HF_ENDPOINT` and the Hugging Face token are read from the environment.

## Library

```rust
use speech2text::whisper::{Transcriber, default_device, hub};

let checkpoint = hub::checkpoint_or_default("large-v3-turbo");
let dir = match hub::cached(checkpoint) {
    Some(dir) => dir,
    None => hub::fetch(checkpoint, &mut |p| eprintln!("{} {}/{}", p.file, p.downloaded, p.total))?,
};
let transcriber = Transcriber::load(&dir, default_device())?;
transcriber.warm_up()?;

let samples: Vec<f32> = /* 16 kHz mono */;
let transcript = transcriber.transcribe(&samples)?;
if !transcript.is_empty() {
    println!("{} ({:.2})", transcript.text, transcript.confidence());
}
```

- `audio::capture::Recorder` records from an input device (by name, or the
  default one), reports the live level, and `Recording::into_16k` gives Whisper-ready samples.
- `audio::resample::resample` and `audio::downmix` convert other audio to 16 kHz mono.
- `audio::is_silent` / `audio::peak_rms` catch empty recordings before they reach the model.
- `hub::fetch_unless` downloads with a cancellation check that can stop in the
  middle of a file. What was already downloaded is kept, so the next try resumes.

Transcription is English only. Audio that Whisper judges to hold no speech
comes back as an empty transcript, and so do the phrases it usually invents for
silence or room noise ("thank you", "thanks for watching", ...).

## Building

Rust 1.88 or later. On Linux, cpal needs the ALSA headers
(`apt install libasound2-dev`). Dependencies are optimized in debug builds too
(`[profile.dev.package."*"]`), since unoptimized candle is unusably slow.

### CUDA

```sh
cargo run --release --features cuda --example transcribe -- tests/data/buy_milk.wav
```

The `cuda` feature needs the CUDA toolkit (`nvcc`) at build time; the kernels
are compiled for the GPU that `nvidia-smi` reports, or for `$CUDA_COMPUTE_CAP`
(e.g. `120`) on a machine without one. `default_device()` then picks the first
CUDA GPU, falling back to the CPU when none can be opened. Things that can get
in the way:

- The driver must be at least as new as the toolkit. Kernels built by a newer
  `nvcc` than the driver's CUDA version (the one `nvidia-smi` prints) fail at
  warm-up with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
- cudarc only accepts toolkit versions it knows. For a newer one it stops with
  `Unsupported cuda toolkit version`; set `CUDARC_CUDA_VERSION` to the newest
  it supports, e.g. `13040` for 13.4.
- On Windows, build from a Visual Studio developer prompt (or after running
  `vcvars64.bat`) so `nvcc` finds `cl.exe`. CUDA 13 also needs the conforming
  preprocessor: `set NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor`.

## Tests

```sh
cargo test                        # unit tests
cargo test --release -- --ignored # end to end, downloads both checkpoints (~2.6 GB)
cargo test --release --features cuda -- --ignored # the same on an NVIDIA GPU
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
