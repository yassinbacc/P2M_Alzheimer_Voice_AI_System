import 'dart:io';
import 'dart:typed_data';

import 'wav_duration.dart';

Future<bool> audioFileExists(String path) async {
  return File(path).exists();
}

Future<void> deleteAudioFile(String path) async {
  final file = File(path);
  if (await file.exists()) {
    await file.delete();
  }
}

Future<Duration?> estimateWavDuration(String path) async {
  try {
    final file = File(path);
    if (!await file.exists()) return null;
    final bytes = await file.readAsBytes();
    return estimateWavDurationFromBytes(Uint8List.fromList(bytes));
  } catch (_) {
    return null;
  }
}
