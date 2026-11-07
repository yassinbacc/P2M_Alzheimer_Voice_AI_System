import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:web/web.dart' as web;

/// Reads the real duration of an in-memory audio blob (WebM, MP4, WAV, …).
Future<double?> audioBlobDurationSeconds(
  Uint8List bytes, {
  String mimeType = 'audio/webm',
}) async {
  if (bytes.isEmpty) return null;

  final blob = web.Blob(
    <JSUint8Array>[bytes.toJS].toJS,
    web.BlobPropertyBag(type: mimeType),
  );
  final url = web.URL.createObjectURL(blob);
  final audio = web.HTMLAudioElement()..src = url;

  final completer = Completer<double?>();

  void finish(double? value) {
    if (!completer.isCompleted) completer.complete(value);
    web.URL.revokeObjectURL(url);
    audio.src = '';
  }

  audio.onloadedmetadata = ((web.Event _) {
    final d = audio.duration;
    if (d.isFinite && d > 0) {
      finish(d);
    } else {
      finish(null);
    }
  }).toJS;

  audio.onerror = ((web.Event _) {
    debugPrint('web_blob_duration: metadata error');
    finish(null);
  }).toJS;

  try {
    audio.load();
  } catch (e) {
    finish(null);
  }

  return completer.future.timeout(
    const Duration(seconds: 8),
    onTimeout: () {
      debugPrint('web_blob_duration: timeout');
      finish(null);
      return null;
    },
  );
}
