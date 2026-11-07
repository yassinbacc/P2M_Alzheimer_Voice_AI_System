import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import '../models/assessment.dart';
import '../models/patient.dart';
import '../utils/api_config.dart';

class ApiService {
  late Dio _dio;
  final String baseUrl;

  ApiService({String? baseUrl}) : baseUrl = baseUrl ?? ApiConfig.baseUrl {
    _dio = Dio(
      BaseOptions(
        baseUrl: this.baseUrl,
        connectTimeout: ApiConfig.defaultTimeout,
        receiveTimeout: ApiConfig.defaultTimeout,
        contentType: 'application/json',
      ),
    );

  }

  Future<List<Patient>> getAllPatients() async {
    final response = await _dio.get('/patients');
    return (response.data as List)
        .map((p) => Patient.fromJson(p as Map<String, dynamic>))
        .toList();
  }

  Future<Patient> getPatient(String id) async {
    final response = await _dio.get('/patients/$id');
    return Patient.fromJson(response.data as Map<String, dynamic>);
  }

  Future<Patient> createPatient(Patient patient) async {
    final response = await _dio.post('/patients', data: patient.toJson());
    return Patient.fromJson(response.data as Map<String, dynamic>);
  }

  Future<Patient> updatePatient(String id, Patient patient) async {
    final response = await _dio.put('/patients/$id', data: patient.toJson());
    return Patient.fromJson(response.data as Map<String, dynamic>);
  }

  Future<void> deletePatient(String id) async {
    await _dio.delete('/patients/$id');
  }

  Future<List<Assessment>> getPatientAssessments(String patientId) async {
    final response = await _dio.get(
      '/assessments',
      queryParameters: {'patientId': patientId},
    );
    return (response.data as List)
        .map((a) => Assessment.fromJson(a as Map<String, dynamic>))
        .toList();
  }

  Future<Assessment> getAssessment(String id) async {
    final response = await _dio.get('/assessments/$id');
    return Assessment.fromJson(response.data as Map<String, dynamic>);
  }

  /// Analyze recorded or uploaded audio (Whisper + model — may take several minutes).
  ///
  /// On web, pass [audioBytes] and [audioFilename] (path alone is not usable).
  Future<Assessment> analyzeAudio(
    String patientId,
    String audioFilePath, {
    String? preset,
    List<int>? audioBytes,
    String? audioFilename,
  }) async {
    final MultipartFile audioPart;
    if (audioBytes != null && audioBytes.isNotEmpty) {
      audioPart = MultipartFile.fromBytes(
        audioBytes,
        filename: audioFilename ?? 'audio.wav',
      );
    } else {
      audioPart = await MultipartFile.fromFile(
        audioFilePath,
        filename: audioFilePath.split(RegExp(r'[/\\]')).last,
      );
    }

    final formData = FormData.fromMap({
      'patientId': patientId,
      'audio': audioPart,
      if (preset != null) 'preset': preset,
    });

    final analyzeDio = Dio(
      BaseOptions(
        baseUrl: baseUrl,
        connectTimeout: ApiConfig.analyzeTimeout,
        receiveTimeout: ApiConfig.analyzeTimeout,
        sendTimeout: ApiConfig.analyzeTimeout,
      ),
    );

    try {
      final response = await analyzeDio.post(
        '/assessments/analyze',
        data: formData,
      );
      return Assessment.fromJson(response.data as Map<String, dynamic>);
    } on DioException catch (e) {
      final detail = e.response?.data;
      if (detail is Map && detail['detail'] != null) {
        throw Exception(detail['detail'].toString());
      }
      rethrow;
    }
  }

  Future<Assessment> saveAssessment(Assessment assessment) async {
    final response = await _dio.post('/assessments', data: assessment.toJson());
    return Assessment.fromJson(response.data as Map<String, dynamic>);
  }

  Future<void> deleteAssessment(String id) async {
    await _dio.delete('/assessments/$id');
  }

  Future<bool> healthCheck() async {
    try {
      final response = await _dio.get('/health');
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
