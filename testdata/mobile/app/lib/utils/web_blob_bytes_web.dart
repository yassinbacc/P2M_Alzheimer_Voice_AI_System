import 'dart:typed_data';

import 'package:cross_file/cross_file.dart';
import 'package:flutter/foundation.dart';

/// Read bytes from a `blob:` URL returned by record_web after [AudioRecorder.stop].
Future<Uint8List?> readBlobUrlBytes(String? blobUrl) async {
  if (blobUrl == null || blobUrl.isEmpty) return null;
  try {
    final bytes = await XFile(blobUrl).readAsBytes();
    return bytes.isEmpty ? null : bytes;
  } catch (e, st) {
    debugPrint('web_blob read failed: $e\n$st');
    return null;
  }
}
