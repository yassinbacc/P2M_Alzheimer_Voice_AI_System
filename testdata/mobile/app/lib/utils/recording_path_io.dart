import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';

Future<String?> createRecordingFilePath() async {
  final dir = await getApplicationDocumentsDirectory();
  final folder = Directory('${dir.path}/audio_recordings');
  if (!await folder.exists()) {
    await folder.create(recursive: true);
  }
  final fileName = 'recording_${DateTime.now().millisecondsSinceEpoch}.wav';
  return '${folder.path}/$fileName';
}

Future<Uint8List?> readRecordingBytes(String path) async {
  final file = File(path);
  if (!await file.exists()) return null;
  return file.readAsBytes();
}
