import 'patient.dart';
import 'assessment.dart';

class AppState {
  final Patient? currentPatient;
  final List<Assessment> assessments;
  final bool isLoading;
  final String? errorMessage;
  final bool isRecording;

  AppState({
    this.currentPatient,
    this.assessments = const [],
    this.isLoading = false,
    this.errorMessage,
    this.isRecording = false,
  });

  AppState copyWith({
    Patient? currentPatient,
    List<Assessment>? assessments,
    bool? isLoading,
    String? errorMessage,
    bool? isRecording,
  }) {
    return AppState(
      currentPatient: currentPatient ?? this.currentPatient,
      assessments: assessments ?? this.assessments,
      isLoading: isLoading ?? this.isLoading,
      errorMessage: errorMessage ?? this.errorMessage,
      isRecording: isRecording ?? this.isRecording,
    );
  }
}
