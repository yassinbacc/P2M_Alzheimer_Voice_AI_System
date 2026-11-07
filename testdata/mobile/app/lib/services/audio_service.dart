import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';

import '../utils/audio_file_helper.dart';
import '../utils/recording_path.dart';
import '../utils/wav_duration.dart';
import '../utils/web_blob_bytes.dart';
import '../utils/web_blob_duration.dart';
import 'web_audio_capture.dart';

/// Result of a completed recording (file path and/or in-memory WAV bytes).
class RecordingCapture {
  final String? path;
  final Uint8List? bytes;
  final String displayName;
  /// PCM/container duration in seconds (from file bytes, not UI stopwatch).
  final double durationSec;

  const RecordingCapture({
    this.path,
    this.bytes,
    required this.displayName,
    required this.durationSec,
  });

  bool get hasAudio =>
      (bytes != null && bytes!.isNotEmpty) || (path != null && path!.isNotEmpty);
}

class AudioService {
  static const int sampleRate = 16000;
  /// WAV with only a header is ~44 B; real speech is much larger.
  static const int minWavBytes = 2000;

  final AudioRecorder _recorder = AudioRecorder();
  String? _recordingPath;
  bool _isRecording = false;

  WebAudioCapture? _webCapture;
  Timer? _webProgressTimer;
  DateTime? _webRecordStarted;

  /// Live capture progress on web (updates ~4×/s while recording).
  final ValueNotifier<double> webCaptureSeconds = ValueNotifier(0);

  bool get isRecording => _isRecording;

  RecordConfig get _nativeWavConfig => const RecordConfig(
        encoder: AudioEncoder.wav,
        bitRate: 128000,
        sampleRate: sampleRate,
        numChannels: 1,
      );

  Future<bool> requestPermissions() async {
    try {
      return await _recorder.hasPermission();
    } catch (e) {
      debugPrint('AudioService: permission error: $e');
      return false;
    }
  }

  Future<bool> isEncoderSupported() async {
    if (kIsWeb) {
      final opus = await _recorder.isEncoderSupported(AudioEncoder.opus);
      final pcm = await _recorder.isEncoderSupported(AudioEncoder.pcm16bits);
      return opus || pcm;
    }
    return _recorder.isEncoderSupported(AudioEncoder.wav);
  }

  Future<Amplitude> getAmplitude() => _recorder.getAmplitude();

  Stream<Amplitude> onAmplitudeChanged(Duration interval) =>
      _recorder.onAmplitudeChanged(interval);

  Future<String?> startRecording() async {
    if (_isRecording) {
      debugPrint('AudioService: already recording');
      return null;
    }
    final ok = await _startRecordingInternal();
    return ok ? _recordingPath : null;
  }

  Future<bool> _startRecordingInternal() async {
    try {
      if (!await requestPermissions()) {
        debugPrint('AudioService: microphone permission denied');
        return false;
      }

      if (!await isEncoderSupported()) {
        debugPrint('AudioService: no supported web/native encoder');
        return false;
      }

      if (kIsWeb) {
        return _startWebRecording();
      }

      final path = await createRecordingFilePath();
      if (path == null) {
        debugPrint('AudioService: could not create recording path');
        return false;
      }
      _recordingPath = path;
      await _recorder.start(_nativeWavConfig, path: path);
      _isRecording = true;
      return true;
    } catch (e, st) {
      debugPrint('AudioService: start failed: $e\n$st');
      _isRecording = false;
      return false;
    }
  }

  /// Web: MediaRecorder (WebM/opus) — not the AudioWorklet WAV path, which truncates.
  Future<bool> _startWebRecording() async {
    _webCapture = WebAudioCapture(_recorder);
    final ok = await _webCapture!.start();
    if (!ok) {
      _webCapture = null;
      return false;
    }

    _recordingPath = 'recording.webm';
    _isRecording = true;
    _webRecordStarted = DateTime.now();
    webCaptureSeconds.value = 0;

    _webProgressTimer?.cancel();
    _webProgressTimer = Timer.periodic(const Duration(milliseconds: 250), (_) {
      final cap = _webCapture;
      final t0 = _webRecordStarted;
      if (cap == null || !cap.isActive || t0 == null) return;
      webCaptureSeconds.value = cap.capturedBytes > 0
          ? cap.estimatedCaptureSec
          : DateTime.now().difference(t0).inMilliseconds / 1000.0;
    });

    return true;
  }

  Future<RecordingCapture?> stopRecording() async {
    if (!_isRecording) return null;

    try {
      if (kIsWeb) {
        return _stopWebRecording();
      }

      final blobOrPath = await _recorder.stop();
      _isRecording = false;

      final path = blobOrPath ?? _recordingPath;
      if (path == null) return null;

      _recordingPath = path;
      final name = path.split(RegExp(r'[/\\]')).last;
      final bytes = await readRecordingBytes(path);

      final capture = _finalizeCapture(
        path: path,
        bytes: bytes,
        displayName: name,
      );
      if (capture == null) {
        debugPrint('AudioService: native WAV invalid ($path)');
      }
      return capture;
    } catch (e, st) {
      debugPrint('AudioService: stop failed: $e\n$st');
      _isRecording = false;
      return null;
    }
  }

  Future<RecordingCapture?> _stopWebRecording() async {
    _webProgressTimer?.cancel();
    _webProgressTimer = null;
    _webRecordStarted = null;

    final capture = _webCapture;
    _webCapture = null;
    _isRecording = false;

    if (capture == null) return null;

    final raw = await capture.stop();
    if (raw == null) return null;

    debugPrint(
      'AudioService: web raw ${raw.bytes.length} bytes (${raw.filename})',
    );

    double? durationSec = await audioBlobDurationSeconds(
      raw.bytes,
      mimeType: raw.mime,
    );

    if (raw.filename.endsWith('.wav')) {
      final fixed = repairWavHeader(raw.bytes);
      durationSec ??= wavDurationSeconds(fixed);
      return _finalizeCapture(
        path: raw.filename,
        bytes: fixed,
        displayName: raw.filename,
        durationSec: durationSec,
      );
    }

    durationSec ??= wavDurationSeconds(raw.bytes);
    if (durationSec == null || durationSec <= 0) {
      debugPrint('AudioService: could not measure webm/wav duration');
      return null;
    }

    webCaptureSeconds.value = durationSec;

    return RecordingCapture(
      path: raw.filename,
      bytes: raw.bytes,
      displayName: raw.filename,
      durationSec: durationSec,
    );
  }

  RecordingCapture? _finalizeCapture({
    required String? path,
    required Uint8List? bytes,
    required String displayName,
    double? durationSec,
  }) {
    if (bytes == null || bytes.length < minWavBytes) return null;
    if (!_isValidWav(bytes)) return null;

    final fixed = repairWavHeader(bytes);
    final sec = durationSec ?? wavDurationSeconds(fixed);
    if (sec == null || sec <= 0) {
      debugPrint('AudioService: could not measure WAV duration');
      return null;
    }

    debugPrint(
      'AudioService: capture ${fixed.length} bytes, ${sec.toStringAsFixed(1)}s',
    );

    return RecordingCapture(
      path: path,
      bytes: fixed,
      displayName: displayName,
      durationSec: sec,
    );
  }

  static bool _isValidWav(Uint8List? bytes) {
    if (bytes == null || bytes.length < minWavBytes) return false;
    if (bytes.length < 12) return false;
    final riff = String.fromCharCodes(bytes.sublist(0, 4));
    final wave = String.fromCharCodes(bytes.sublist(8, 12));
    return riff == 'RIFF' && wave == 'WAVE';
  }

  Future<void> cancelRecording() async {
    try {
      _webProgressTimer?.cancel();
      _webProgressTimer = null;
      _webRecordStarted = null;
      webCaptureSeconds.value = 0;

      if (kIsWeb) {
        await _webCapture?.cancel();
        _webCapture = null;
      } else if (_isRecording) {
        await _recorder.cancel();
      }
      if (_recordingPath != null && !kIsWeb) {
        await deleteAudioFile(_recordingPath!);
      }
      _isRecording = false;
      _recordingPath = null;
    } catch (e) {
      debugPrint('AudioService: cancel failed: $e');
      _isRecording = false;
    }
  }

  Future<Duration?> getRecordingDuration(String filePath) async {
    return estimateWavDuration(filePath);
  }

  Future<void> deleteRecording(String filePath) async {
    await deleteAudioFile(filePath);
  }

  void dispose() {
    _webProgressTimer?.cancel();
    webCaptureSeconds.dispose();
    _recorder.dispose();
  }
}
