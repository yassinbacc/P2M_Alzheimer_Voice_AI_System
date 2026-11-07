import 'dart:typed_data';

/// Wrap raw PCM16 mono samples in a minimal WAV container (for web recording).
Uint8List pcm16ToWav(
  Uint8List pcm, {
  int sampleRate = 16000,
  int channels = 1,
}) {
  const bitsPerSample = 16;
  final byteRate = sampleRate * channels * bitsPerSample ~/ 8;
  final blockAlign = channels * bitsPerSample ~/ 8;
  final dataSize = pcm.length;

  final header = ByteData(44);
  _tag(header, 0, 'RIFF');
  header.setUint32(4, 36 + dataSize, Endian.little);
  _tag(header, 8, 'WAVE');
  _tag(header, 12, 'fmt ');
  header.setUint32(16, 16, Endian.little);
  header.setUint16(20, 1, Endian.little);
  header.setUint16(22, channels, Endian.little);
  header.setUint32(24, sampleRate, Endian.little);
  header.setUint32(28, byteRate, Endian.little);
  header.setUint16(32, blockAlign, Endian.little);
  header.setUint16(34, bitsPerSample, Endian.little);
  _tag(header, 36, 'data');
  header.setUint32(40, dataSize, Endian.little);

  final out = Uint8List(44 + dataSize);
  out.setAll(0, header.buffer.asUint8List(0, 44));
  out.setAll(44, pcm);
  return out;
}

void _tag(ByteData data, int offset, String tag) {
  for (var i = 0; i < tag.length; i++) {
    data.setUint8(offset + i, tag.codeUnitAt(i));
  }
}
