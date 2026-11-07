import 'dart:convert';

import 'package:flutter/foundation.dart';

import '../models/patient.dart';
import 'app_preferences.dart';
import 'assessment_repository.dart';
import 'local_db.dart';

/// Restore / persist logged-in patient across app restarts.
/// On web, uses SharedPreferences only (sqflite is not available).
class SessionService {
  final AppPreferences _prefs = AppPreferences();
  final LocalDb _db = LocalDb();

  Future<void> savePatient(Patient patient) async {
    await _prefs.savePatientJson(jsonEncode(patient.toJson()));
    if (kIsWeb) return;
    try {
      await _db.insertPatient(patient);
    } catch (e) {
      debugPrint('SessionService: DB save skipped/failed: $e');
    }
  }

  Future<Patient?> loadPatient() async {
    final raw = await _prefs.loadPatientJson();
    if (raw == null) return null;
    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      if (!kIsWeb) {
        final fromDb = await _db.getPatient(map['id'] as String);
        if (fromDb != null) return fromDb;
      }
      return Patient.fromJson(map);
    } catch (e) {
      debugPrint('SessionService: loadPatient failed: $e');
      return null;
    }
  }

  Future<void> clearAll() async {
    await _prefs.clearSession();
    await AssessmentRepository().clearAll();
    if (kIsWeb) return;
    try {
      await _db.clearPatients();
    } catch (e) {
      debugPrint('SessionService: clearAll DB failed: $e');
    }
  }
}
