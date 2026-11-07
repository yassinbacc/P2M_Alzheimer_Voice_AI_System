import '../utils/audio_validation_constants.dart';
import '../widgets/verification_checklist.dart';

/// Pre-analysis checks shown in the recording screen Vérification tab.
List<VerificationItem> buildRecordingVerification({
  required bool micOk,
  required bool hasClip,
  required double durationSec,
  required int? fileBytes,
  required bool isRecording,
}) {
  final minSec = AudioValidationConstants.minDurationSeconds;
  final bytes = fileBytes ?? 0;

  return [
    VerificationItem(
      label: 'Microphone autorisé',
      detail: micOk ? 'Accès micro OK' : 'Autorisez le micro dans le navigateur',
      passed: micOk,
    ),
    VerificationItem(
      label: 'Fichier audio présent',
      detail: hasClip ? 'Prêt pour l’envoi' : 'Enregistrez ou importez un fichier',
      passed: hasClip,
    ),
    VerificationItem(
      label: 'Durée minimale ($minSec s)',
      detail: hasClip
          ? 'Durée audio : ${durationSec.toStringAsFixed(1)} s (fichier enregistré)'
          : 'Parlez en continu pendant la description',
      passed: hasClip && durationSec >= minSec,
    ),
    VerificationItem(
      label: 'Taille du fichier',
      detail: hasClip ? '$bytes octets' : '—',
      passed: hasClip && bytes >= 1000,
    ),
    VerificationItem(
      label: 'Enregistrement terminé',
      detail: isRecording ? 'Arrêtez avant d’analyser' : 'Aucun enregistrement actif',
      passed: !isRecording,
    ),
  ];
}

bool recordingReadyForAnalysis(List<VerificationItem> items) {
  return items.where((i) => i.required).every((i) => i.passed);
}
