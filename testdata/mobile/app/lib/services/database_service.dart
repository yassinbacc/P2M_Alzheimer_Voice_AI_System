import 'dart:convert';

import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';

import '../models/assessment.dart';
import '../models/patient.dart';

class DatabaseService {
  static final DatabaseService _instance = DatabaseService._internal();
  static Database? _database;

  factory DatabaseService() => _instance;

  DatabaseService._internal();

  Future<Database> get database async {
    _database ??= await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'alzheimer_detection.db');

    return openDatabase(
      path,
      version: 2,
      onCreate: _createTables,
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          await db.execute('DROP TABLE IF EXISTS assessments');
          await db.execute('''
            CREATE TABLE assessments(
              id TEXT PRIMARY KEY,
              patientId TEXT NOT NULL,
              audioFilePath TEXT NOT NULL,
              recordedAt TEXT NOT NULL,
              duration INTEGER NOT NULL,
              transcription TEXT,
              riskScore REAL NOT NULL,
              riskLevel TEXT NOT NULL,
              confidence REAL NOT NULL,
              biomarkers TEXT,
              analysisDetails TEXT,
              notes TEXT,
              createdAt TEXT NOT NULL
            )
          ''');
          await db.execute(
            'CREATE INDEX idx_assessments_patientId ON assessments(patientId)',
          );
        }
      },
    );
  }

  Future<void> _createTables(Database db, int version) async {
    await db.execute('''
      CREATE TABLE patients(
        id TEXT PRIMARY KEY,
        firstName TEXT NOT NULL,
        lastName TEXT NOT NULL,
        dateOfBirth TEXT NOT NULL,
        gender TEXT NOT NULL,
        email TEXT NOT NULL,
        phoneNumber TEXT NOT NULL,
        medicalHistory TEXT,
        createdAt TEXT NOT NULL,
        updatedAt TEXT NOT NULL
      )
    ''');

    await db.execute('''
      CREATE TABLE assessments(
        id TEXT PRIMARY KEY,
        patientId TEXT NOT NULL,
        audioFilePath TEXT NOT NULL,
        recordedAt TEXT NOT NULL,
        duration INTEGER NOT NULL,
        transcription TEXT,
        riskScore REAL NOT NULL,
        riskLevel TEXT NOT NULL,
        confidence REAL NOT NULL,
        biomarkers TEXT,
        analysisDetails TEXT,
        notes TEXT,
        createdAt TEXT NOT NULL
      )
    ''');

    await db.execute(
      'CREATE INDEX idx_assessments_patientId ON assessments(patientId)',
    );
    await db.execute(
      'CREATE INDEX idx_assessments_recordedAt ON assessments(recordedAt)',
    );
  }

  Future<void> insertPatient(Patient patient) async {
    final db = await database;
    await db.insert(
      'patients',
      _patientToMap(patient),
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<Patient?> getPatient(String id) async {
    final db = await database;
    final maps = await db.query('patients', where: 'id = ?', whereArgs: [id]);
    if (maps.isEmpty) return null;
    return _mapToPatient(maps.first);
  }

  Future<void> updatePatient(Patient patient) async {
    final db = await database;
    await db.update(
      'patients',
      _patientToMap(patient),
      where: 'id = ?',
      whereArgs: [patient.id],
    );
  }

  Future<void> insertAssessment(Assessment assessment) async {
    final db = await database;
    await db.insert(
      'assessments',
      _assessmentToMap(assessment),
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<List<Assessment>> getPatientAssessments(String patientId) async {
    final db = await database;
    final maps = await db.query(
      'assessments',
      where: 'patientId = ?',
      whereArgs: [patientId],
      orderBy: 'recordedAt DESC',
    );
    return maps.map(_mapToAssessment).toList();
  }

  Future<Assessment?> getAssessment(String id) async {
    final db = await database;
    final maps = await db.query('assessments', where: 'id = ?', whereArgs: [id]);
    if (maps.isEmpty) return null;
    return _mapToAssessment(maps.first);
  }

  Future<void> deleteAssessment(String id) async {
    final db = await database;
    await db.delete('assessments', where: 'id = ?', whereArgs: [id]);
  }

  Future<int> countAssessments(String patientId) async {
    final db = await database;
    final r = await db.rawQuery(
      'SELECT COUNT(*) as c FROM assessments WHERE patientId = ?',
      [patientId],
    );
    return Sqflite.firstIntValue(r) ?? 0;
  }

  Map<String, dynamic> _patientToMap(Patient patient) => {
        'id': patient.id,
        'firstName': patient.firstName,
        'lastName': patient.lastName,
        'dateOfBirth': patient.dateOfBirth.toIso8601String(),
        'gender': patient.gender,
        'email': patient.email,
        'phoneNumber': patient.phoneNumber,
        'medicalHistory': patient.medicalHistory,
        'createdAt': patient.createdAt.toIso8601String(),
        'updatedAt': patient.updatedAt.toIso8601String(),
      };

  Patient _mapToPatient(Map<String, dynamic> map) => Patient(
        id: map['id'] as String,
        firstName: map['firstName'] as String,
        lastName: map['lastName'] as String,
        dateOfBirth: DateTime.parse(map['dateOfBirth'] as String),
        gender: map['gender'] as String,
        email: map['email'] as String? ?? '',
        phoneNumber: map['phoneNumber'] as String? ?? '',
        medicalHistory: map['medicalHistory'] as String? ?? '',
        createdAt: DateTime.parse(map['createdAt'] as String),
        updatedAt: DateTime.parse(map['updatedAt'] as String),
      );

  Map<String, dynamic> _assessmentToMap(Assessment assessment) => {
        'id': assessment.id,
        'patientId': assessment.patientId,
        'audioFilePath': assessment.audioFilePath,
        'recordedAt': assessment.recordedAt.toIso8601String(),
        'duration': assessment.duration.inSeconds,
        'transcription': assessment.transcription,
        'riskScore': assessment.riskScore.score,
        'riskLevel': assessment.riskScore.riskLevel,
        'confidence': assessment.riskScore.confidence,
        'biomarkers': jsonEncode(assessment.riskScore.biomarkers),
        'analysisDetails': jsonEncode(assessment.analysisDetails),
        'notes': assessment.notes,
        'createdAt': assessment.createdAt.toIso8601String(),
      };

  Assessment _mapToAssessment(Map<String, dynamic> map) {
    List<String> biomarkers = [];
    final bioRaw = map['biomarkers'] as String?;
    if (bioRaw != null && bioRaw.isNotEmpty) {
      try {
        biomarkers = List<String>.from(jsonDecode(bioRaw) as List);
      } catch (_) {
        biomarkers = bioRaw.split('|');
      }
    }

    Map<String, dynamic> details = {};
    final detRaw = map['analysisDetails'] as String?;
    if (detRaw != null && detRaw.isNotEmpty) {
      try {
        details = Map<String, dynamic>.from(jsonDecode(detRaw) as Map);
      } catch (_) {
        details = {};
      }
    }

    return Assessment(
      id: map['id'] as String,
      patientId: map['patientId'] as String,
      audioFilePath: map['audioFilePath'] as String,
      recordedAt: DateTime.parse(map['recordedAt'] as String),
      duration: Duration(seconds: map['duration'] as int),
      transcription: map['transcription'] as String? ?? '',
      riskScore: AlzheimersRiskScore(
        score: (map['riskScore'] as num).toDouble(),
        riskLevel: map['riskLevel'] as String,
        confidence: (map['confidence'] as num).toDouble(),
        biomarkers: biomarkers,
      ),
      analysisDetails: details,
      notes: map['notes'] as String? ?? '',
      createdAt: DateTime.parse(map['createdAt'] as String),
    );
  }
}
