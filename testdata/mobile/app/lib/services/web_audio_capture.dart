import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';

import '../utils/wav_encoder.dart';
import '../utils/web_blob_bytes.dart';
import '../utils/web_blob_duration.dart';

/// Web-only capture that avoids the WAV AudioWorklet path (truncates when the tab sleeps).
class WebAudioCapture {
  final AudioRecorder _recorder;

  StreamSubscription<Uint8List>? _pcmSub;
  final BytesBuilder _pcm = BytesBuilder(copy: false);
  Completer<void>? _pcmDrain;

  bool _useMediaRecorder = false;
  bool _active = false;
  int _mediaChunkBytes = 0;

  WebAudioCapture(this._recorder);

  bool get isActive => _active;

  /// Bytes captured so far (PCM or compressed chunks).
  int get capturedBytes => _useMediaRecorder ? _mediaChunkBytes : _pcm.length;

  /// Rough live duration for UI — PCM byte count / 16 kHz mono 16-bit.
  double get estimatedCaptureSec => capturedBytes / (16000 * 2);

  Future<bool> start() async {
    if (_active) return false;

    _pcm.clear();
    _mediaChunkBytes = 0;

    if (await _recorder.isEncoderSupported(AudioEncoder.opus)) {
      return _startMediaRecorder();
    }
    return _startPcmStream();
  }

  /// [MediaRecorder] — reliable for long recordings in Chrome.
  Future<bool> _startMediaRecorder() async {
    const config = RecordConfig(
      encoder: AudioEncoder.opus,
      bitRate: 128000,
      sampleRate: 16000,
      numChannels: 1,
      autoGain: true,
      echoCancel: true,
      noiseSuppress: true,
    );

    try {
      await _recorder.start(config, path: '');
      _useMediaRecorder = true;
      _active = true;
      debugPrint('WebAudioCapture: started MediaRecorder (opus/webm)');
      return true;
    } catch (e, st) {
      debugPrint('WebAudioCapture: MediaRecorder start failed: $e\n$st');
      return _startPcmStream();
    }
  }

  /// Fallback: accumulate raw PCM and build WAV ourselves.
  Future<bool> _startPcmStream() async {
    const config = RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: 16000,
      numChannels: 1,
      streamBufferSize: 4096,
      autoGain: true,
      echoCancel: true,
      noiseSuppress: true,
    );

    try {
      final stream = await _recorder.startStream(config);
      _pcmDrain = Completer<void>();
      var chunks = 0;

      _pcmSub = stream.listen(
        (chunk) {
          if (chunk.isNotEmpty) {
            _pcm.add(chunk);
            chunks++;
          }
        },
        onDone: () {
          debugPrint(
            'WebAudioCapture: PCM stream done chunks=$chunks bytes=${_pcm.length}',
          );
          if (!(_pcmDrain?.isCompleted ?? true)) {
            _pcmDrain!.complete();
          }
        },
        onError: (e, st) {
          debugPrint('WebAudioCapture: PCM stream error: $e\n$st');
          if (!(_pcmDrain?.isCompleted ?? true)) {
            _pcmDrain!.completeError(e);
          }
        },
        cancelOnError: false,
      );

      _useMediaRecorder = false;
      _active = true;
      debugPrint('WebAudioCapture: started PCM stream');
      return true;
    } catch (e, st) {
      debugPrint('WebAudioCapture: PCM stream start failed: $e\n$st');
      return false;
    }
  }

  Future<({Uint8List bytes, String filename, String mime})?> stop() async {
    if (!_active) return null;
    _active = false;

    try {
      if (_useMediaRecorder) {
        return await _stopMediaRecorder();
      }
      return await _stopPcmStream();
    } catch (e, st) {
      debugPrint('WebAudioCapture: stop failed: $e\n$st');
      return null;
    }
  }

  Future<({Uint8List bytes, String filename, String mime})?> _stopMediaRecorder() async {
    final blobUrl = await _recorder.stop();
    final bytes = await readBlobUrlBytes(blobUrl);
    debugPrint(
      'WebAudioCapture: webm blob url=${blobUrl ?? "null"} bytes=${bytes?.length ?? 0}',
    );
    if (bytes == null || bytes.length < 1000) return null;

    return (
      bytes: bytes,
      filename: 'recording.webm',
      mime: 'audio/webm',
    );
  }

  Future<({Uint8List bytes, String filename, String mime})?> _stopPcmStream() async {
    await _recorder.stop();
    try {
      await _pcmDrain?.future.timeout(const Duration(seconds: 8));
    } catch (e) {
      debugPrint('WebAudioCapture: PCM drain timeout: $e');
    }
    await _pcmSub?.cancel();
    _pcmSub = null;

    if (_pcm.length < 2000) {
      debugPrint('WebAudioCapture: PCM too small (${_pcm.length} bytes)');
      return null;
    }

    final wav = pcm16ToWav(_pcm.toBytes(), sampleRate: 16000);
    return (bytes: wav, filename: 'recording.wav', mime: 'audio/wav');
  }

  Future<void> cancel() async {
    _active = false;
    await _pcmSub?.cancel();
    _pcmSub = null;
    _pcm.clear();
    _mediaChunkBytes = 0;
    try {
      await _recorder.cancel();
    } catch (e) {
      debugPrint('WebAudioCapture: cancel $e');
    }
  }
}
