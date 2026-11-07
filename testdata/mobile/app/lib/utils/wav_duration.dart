import 'dart:typed_data';

/// PCM layout parsed from a RIFF/WAVE file.
class WavPcmInfo {
  final int sampleRate;
  final int channels;
  final int bitsPerSample;
  final int pcmDataOffset;
  /// PCM bytes actually present from [pcmDataOffset] to EOF.
  final int pcmByteLength;
  /// Size field written in the `data` chunk header (often wrong on web).
  final int declaredDataSize;
  final int dataChunkSizeOffset;

  const WavPcmInfo({
    required this.sampleRate,
    required this.channels,
    required this.bitsPerSample,
    required this.pcmDataOffset,
    required this.pcmByteLength,
    required this.declaredDataSize,
    required this.dataChunkSizeOffset,
  });

  int get blockAlign => channels * (bitsPerSample ~/ 8);
}

/// Parse fmt + data chunks. Uses actual bytes in the file (not only the data header size).
WavPcmInfo? parseWavPcm(Uint8List bytes) {
  if (bytes.length < 44) return null;
  if (String.fromCharCodes(bytes.sublist(0, 4)) != 'RIFF') return null;
  if (String.fromCharCodes(bytes.sublist(8, 12)) != 'WAVE') return null;

  int? sampleRate;
  int? channels;
  int? bitsPerSample;
  int? pcmDataOffset;
  int? dataSizeFromHeader;
  int? dataChunkSizeOffset;

  var i = 12;
  while (i + 8 <= bytes.length) {
    final id = String.fromCharCodes(bytes.sublist(i, i + 4));
    final size = ByteData.sublistView(bytes, i + 4, i + 8).getUint32(0, Endian.little);
    final contentStart = i + 8;
    if (contentStart > bytes.length) break;

    if (id == 'fmt ' && size >= 16 && contentStart + 16 <= bytes.length) {
      channels = ByteData.sublistView(bytes, contentStart + 2, contentStart + 4)
          .getUint16(0, Endian.little);
      sampleRate = ByteData.sublistView(bytes, contentStart + 4, contentStart + 8)
          .getUint32(0, Endian.little);
      bitsPerSample = ByteData.sublistView(bytes, contentStart + 14, contentStart + 16)
          .getUint16(0, Endian.little);
    } else if (id == 'data') {
      pcmDataOffset = contentStart;
      dataSizeFromHeader = size;
      dataChunkSizeOffset = i + 4;
    }

    final padded = size + (size.isOdd ? 1 : 0);
    final next = contentStart + padded;
    if (next <= i) break;
    i = next;
  }

  if (pcmDataOffset == null ||
      sampleRate == null ||
      channels == null ||
      bitsPerSample == null ||
      dataChunkSizeOffset == null) {
    return null;
  }
  if (sampleRate == 0 || channels == 0 || bitsPerSample == 0) return null;

  final available = bytes.length - pcmDataOffset;
  if (available <= 0) return null;

  final declared = dataSizeFromHeader ?? 0;

  return WavPcmInfo(
    sampleRate: sampleRate,
    channels: channels,
    bitsPerSample: bitsPerSample,
    pcmDataOffset: pcmDataOffset,
    pcmByteLength: available,
    declaredDataSize: declared,
    dataChunkSizeOffset: dataChunkSizeOffset,
  );
}

/// Duration in seconds from PCM bytes (same basis as librosa: sample count / sample rate).
double? wavDurationSeconds(Uint8List? bytes) {
  final info = bytes == null ? null : parseWavPcm(bytes);
  if (info == null || info.blockAlign == 0) return null;
  return info.pcmByteLength / (info.sampleRate * info.blockAlign);
}

/// Estimate duration from a WAV file in memory.
Duration? estimateWavDurationFromBytes(Uint8List? bytes) {
  final sec = wavDurationSeconds(bytes);
  if (sec == null) return null;
  return Duration(microseconds: (sec * 1000000).round());
}

/// Rounds to one decimal for UI / checklist (matches server `round(duration, 1)`).
double wavDurationSecondsRounded(Uint8List? bytes) {
  final sec = wavDurationSeconds(bytes);
  if (sec == null) return 0;
  return (sec * 10).roundToDouble() / 10;
}

/// Fix RIFF and `data` chunk sizes so uploaded files decode to full captured audio.
Uint8List repairWavHeader(Uint8List bytes) {
  final info = parseWavPcm(bytes);
  if (info == null) return bytes;

  final actualPcm = bytes.length - info.pcmDataOffset;
  if (actualPcm <= 0) return bytes;

  final out = Uint8List.fromList(bytes);
  final needsDataPatch = info.declaredDataSize != actualPcm;

  if (needsDataPatch) {
    ByteData.sublistView(out, info.dataChunkSizeOffset, info.dataChunkSizeOffset + 4)
        .setUint32(0, actualPcm, Endian.little);
  }

  if (out.length >= 8) {
    ByteData.sublistView(out, 4, 8).setUint32(0, out.length - 8, Endian.little);
  }

  return out;
}
