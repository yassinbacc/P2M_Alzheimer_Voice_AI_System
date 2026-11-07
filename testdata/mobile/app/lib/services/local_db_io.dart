import '../models/assessment.dart';
import '../models/patient.dart';
import 'database_service.dart';

/// SQLite via [DatabaseService] on mobile/desktop.
class LocalDb {
  final DatabaseService _db = DatabaseService();

  Future<void> insertPatient(Patient patient) => _db.insertPatient(patient);

  Future<Patient?> getPatient(String id) => _db.getPatient(id);

  Future<void> updatePatient(Patient patient) => _db.updatePatient(patient);

  Future<void> insertAssessment(Assessment assessment) =>
      _db.insertAssessment(assessment);

  Future<List<Assessment>> getPatientAssessments(String patientId) =>
      _db.getPatientAssessments(patientId);

  Future<void> clearAssessments() async {
    final db = await _db.database;
    await db.delete('assessments');
  }

  Future<void> clearPatients() async {
    final db = await _db.database;
    await db.delete('patients');
  }
}
