class Constants {
  // API Configuration
  static const String apiBaseUrl = 'http://localhost:8000/api';
  static const int apiTimeoutSeconds = 10;

  // Audio Recording
  static const int audioSampleRate = 16000;
  static const int audioBitRate = 128000;
  static const int minRecordingDurationSeconds = 120; // 2 minutes
  static const int maxRecordingDurationSeconds = 300; // 5 minutes

  // Assessment
  static const double lowRiskThreshold = 0.3;
  static const double moderateRiskThreshold = 0.7;

  // Storage
  static const String audioRecordingsFolder = 'audio_recordings';
  static const String databaseName = 'alzheimer_detection.db';
  static const int databaseVersion = 1;

  // UI
  static const double defaultPadding = 16.0;
  static const double defaultBorderRadius = 12.0;
  static const double defaultSpacing = 8.0;
}
