import 'package:intl/intl.dart';

class Assessment {
  final String id;
  final String patientId;
  final String audioFilePath;
  final DateTime recordedAt;
  final Duration duration;
  final String transcription;
  final AlzheimersRiskScore riskScore;
  final Map<String, dynamic> analysisDetails;
  final String notes;
  final DateTime createdAt;

  Assessment({
    required this.id,
    required this.patientId,
    required this.audioFilePath,
    required this.recordedAt,
    required this.duration,
    this.transcription = '',
    required this.riskScore,
    this.analysisDetails = const {},
    this.notes = '',
    required this.createdAt,
  });

  String get formattedDate => DateFormat('dd/MM/yyyy HH:mm').format(recordedAt);
  String get formattedDuration =>
      '${duration.inMinutes}m ${duration.inSeconds % 60}s';

  Map<String, dynamic> toJson() => {
        'id': id,
        'patientId': patientId,
        'audioFilePath': audioFilePath,
        'recordedAt': recordedAt.toIso8601String(),
        'duration': duration.inSeconds,
        'transcription': transcription,
        'riskScore': riskScore.toJson(),
        'analysisDetails': analysisDetails,
        'notes': notes,
        'createdAt': createdAt.toIso8601String(),
      };

  factory Assessment.fromJson(Map<String, dynamic> json) => Assessment(
        id: json['id'] as String,
        patientId: json['patientId'] as String,
        audioFilePath: json['audioFilePath'] as String,
        recordedAt: DateTime.parse(json['recordedAt'] as String),
        duration: Duration(seconds: json['duration'] as int),
        transcription: json['transcription'] as String? ?? '',
        riskScore:
            AlzheimersRiskScore.fromJson(json['riskScore'] as Map<String, dynamic>),
        analysisDetails: json['analysisDetails'] as Map<String, dynamic>? ?? {},
        notes: json['notes'] as String? ?? '',
        createdAt: DateTime.parse(json['createdAt'] as String),
      );

  /// True when server rejected clinical labels (invalid / noisy / no description).
  bool get isInconclusive {
    final status = analysisDetails['validationStatus']?.toString();
    final predicted = analysisDetails['predictedClass']?.toString();
    return status == 'invalid' ||
        predicted == 'Non_concluant' ||
        riskScore.riskLevel.toLowerCase().contains('non concluant');
  }

  String? get validationMessage =>
      analysisDetails['validationMessage']?.toString();

  Assessment copyWith({
    String? id,
    String? patientId,
    String? audioFilePath,
    DateTime? recordedAt,
    Duration? duration,
    String? transcription,
    AlzheimersRiskScore? riskScore,
    Map<String, dynamic>? analysisDetails,
    String? notes,
    DateTime? createdAt,
  }) {
    return Assessment(
      id: id ?? this.id,
      patientId: patientId ?? this.patientId,
      audioFilePath: audioFilePath ?? this.audioFilePath,
      recordedAt: recordedAt ?? this.recordedAt,
      duration: duration ?? this.duration,
      transcription: transcription ?? this.transcription,
      riskScore: riskScore ?? this.riskScore,
      analysisDetails: analysisDetails ?? this.analysisDetails,
      notes: notes ?? this.notes,
      createdAt: createdAt ?? this.createdAt,
    );
  }
}

class AlzheimersRiskScore {
  final double score; // 0.0 to 1.0
  final String riskLevel; // Low, Moderate, High
  final double confidence; // 0.0 to 1.0
  final List<String> biomarkers;

  AlzheimersRiskScore({
    required this.score,
    required this.riskLevel,
    required this.confidence,
    this.biomarkers = const [],
  });

  String get riskDescription {
    if (score < 0.3) {
      return 'Low risk - Normal cognitive patterns detected';
    } else if (score < 0.7) {
      return 'Moderate risk - Some cognitive decline indicators present';
    } else {
      return 'High risk - Significant cognitive decline indicators detected';
    }
  }

  Map<String, dynamic> toJson() => {
        'score': score,
        'riskLevel': riskLevel,
        'confidence': confidence,
        'biomarkers': biomarkers,
      };

  factory AlzheimersRiskScore.fromJson(Map<String, dynamic> json) =>
      AlzheimersRiskScore(
        score: (json['score'] as num).toDouble(),
        riskLevel: json['riskLevel'] as String,
        confidence: (json['confidence'] as num).toDouble(),
        biomarkers: List<String>.from(json['biomarkers'] as List? ?? []),
      );
}
