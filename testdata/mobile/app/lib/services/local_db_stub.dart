import '../models/assessment.dart';
import '../models/patient.dart';

/// No-op SQLite on web (sqflite is not used).
class LocalDb {
  Future<void> insertPatient(Patient patient) async {}

  Future<Patient?> getPatient(String id) async => null;

  Future<void> updatePatient(Patient patient) async {}

  Future<void> insertAssessment(Assessment assessment) async {}

  Future<List<Assessment>> getPatientAssessments(String patientId) async => [];

  Future<void> clearAssessments() async {}

  Future<void> clearPatients() async {}
}
