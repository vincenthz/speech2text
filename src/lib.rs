//! On-device English speech recognition: microphone capture (cpal) and
//! Whisper on candle (CUDA or Metal when available).

pub mod audio;
pub mod whisper;
