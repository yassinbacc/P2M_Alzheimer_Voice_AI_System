import 'api_base_url.dart';

/// API base URL for the ML backend.
///
/// - Web / Windows / iOS simulator / desktop: localhost
/// - Android emulator: 10.0.2.2 maps to host machine
/// - Physical device: flutter run --dart-define=API_BASE_URL=http://192.168.1.10:8000/api
class ApiConfig {
  static String get baseUrl {
    const override = String.fromEnvironment('API_BASE_URL');
    if (override.isNotEmpty) {
      return override.endsWith('/api') ? override : '$override/api';
    }
    return defaultApiBaseUrl();
  }

  static const analyzeTimeout = Duration(minutes: 10);
  static const defaultTimeout = Duration(seconds: 30);
}
