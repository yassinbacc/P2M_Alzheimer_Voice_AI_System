import 'package:dio/dio.dart';

/// Readable message from FastAPI / Dio errors.
String messageFromDio(DioException e) {
  final data = e.response?.data;
  if (data is Map) {
    final detail = data['detail'];
    if (detail is String && detail.isNotEmpty) return detail;
    if (detail is List && detail.isNotEmpty) {
      final parts = <String>[];
      for (final item in detail) {
        if (item is Map && item['msg'] != null) {
          parts.add(item['msg'].toString());
        } else {
          parts.add(item.toString());
        }
      }
      if (parts.isNotEmpty) return parts.join('\n');
    }
  }
  if (data is String && data.isNotEmpty) return data;
  return e.message ?? 'Erreur de communication avec le serveur.';
}

String messageFromError(Object e) {
  if (e is DioException) return messageFromDio(e);
  return e.toString().replaceFirst('Exception: ', '');
}
