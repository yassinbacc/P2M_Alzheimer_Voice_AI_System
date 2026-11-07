import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/assessment.dart';
import '../models/patient.dart';
import 'api_service.dart';
import 'local_db.dart';

/// Saves and loads assessments on SQLite (mobile/desktop) or SharedPreferences (web).
class AssessmentRepository {
  static const _keyPrefix = 'assessments_';

  final LocalDb _db = LocalDb();
  final ApiService _api = ApiService();

  String _storageKey(String patientId) => '$_keyPrefix$patientId';

  Future<void> save(Patient patient, Assessment assessment) async {
    await _saveLocal(patient, assessment);
    try {
      if (await _api.healthCheck()) {
        await _api.saveAssessment(assessment);
      }
    } catch (e) {
      debugPrint('AssessmentRepository: API sync skipped: $e');
    }
  }

  Future<void> _saveLocal(Patient patient, Assessment assessment) async {
    if (kIsWeb) {
      await _saveWeb(patient.id, assessment);
      return;
    }
    await _db.insertPatient(patient);
    await _db.insertAssessment(assessment);
  }

  Future<List<Assessment>> listForPatient(String patientId) async {
    final local = kIsWeb
        ? await _loadWeb(patientId)
        : await _db.getPatientAssessments(patientId);

    try {
      if (await _api.healthCheck()) {
        final remote = await _api.getPatientAssessments(patientId);
        return _mergeById(local, remote);
      }
    } catch (e) {
      debugPrint('AssessmentRepository: API list skipped: $e');
    }
    return local;
  }

  Future<void> clearAll() async {
    if (kIsWeb) {
      final prefs = await SharedPreferences.getInstance();
      final keys = prefs.getKeys().where((k) => k.startsWith(_keyPrefix));
      for (final key in keys) {
        await prefs.remove(key);
      }
      return;
    }
    await _db.clearAssessments();
  }

  Future<void> _saveWeb(String patientId, Assessment assessment) async {
    final prefs = await SharedPreferences.getInstance();
    final list = await _loadWeb(patientId);
    list.removeWhere((a) => a.id == assessment.id);
    list.add(assessment);
    list.sort((a, b) => b.recordedAt.compareTo(a.recordedAt));
    final payload = list.map((a) => a.toJson()).toList();
    await prefs.setString(_storageKey(patientId), jsonEncode(payload));
  }

  Future<List<Assessment>> _loadWeb(String patientId) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_storageKey(patientId));
    if (raw == null || raw.isEmpty) return [];
    try {
      final decoded = jsonDecode(raw) as List;
      return decoded
          .map((e) => Assessment.fromJson(Map<String, dynamic>.from(e as Map)))
          .toList();
    } catch (e) {
      debugPrint('AssessmentRepository: web load failed: $e');
      return [];
    }
  }

  List<Assessment> _mergeById(List<Assessment> a, List<Assessment> b) {
    final map = <String, Assessment>{};
    for (final item in [...a, ...b]) {
      map[item.id] = item;
    }
    final merged = map.values.toList();
    merged.sort((x, y) => y.recordedAt.compareTo(x.recordedAt));
    return merged;
  }
}
