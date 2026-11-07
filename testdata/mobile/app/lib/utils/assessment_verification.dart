import '../models/assessment.dart';
import '../widgets/verification_checklist.dart';

/// Post-analysis checks from server [analysisDetails].
List<VerificationItem> buildAssessmentVerification(Assessment a) {
  final d = a.analysisDetails;
  final valid = d['validationStatus']?.toString() == 'valid';
  final code = d['validationCode']?.toString() ?? '—';
  final vd = d['validationDetails'];
  final details = vd is Map ? Map<String, dynamic>.from(vd) : <String, dynamic>{};

  final duration = (details['durationSec'] as num?)?.toDouble() ??
      a.duration.inSeconds.toDouble();
  final words = details['wordCount'] as int?;
  final wpm = details['wordsPerMinute'] as num?;
  final ct = details['ctContentScore'] as num?;

  return [
    VerificationItem(
      label: 'Statut validation serveur',
      detail: valid ? 'Valide ($code)' : 'Non valide ($code)',
      passed: valid,
    ),
    VerificationItem(
      label: 'Durée audio',
      detail: '${duration.toStringAsFixed(0)} s',
      passed: duration >= 30,
    ),
    if (words != null)
      VerificationItem(
        label: 'Mots détectés (Whisper)',
        detail: '$words mots',
        passed: words >= 25,
      ),
    if (wpm != null)
      VerificationItem(
        label: 'Débit de parole',
        detail: '${wpm.toStringAsFixed(0)} mots/min',
        passed: wpm >= 30,
      ),
    if (ct != null)
      VerificationItem(
        label: 'Contenu image (Cookie Theft)',
        detail: 'Score ${ct.toStringAsFixed(2)}',
        passed: ct >= 0.12,
      ),
    VerificationItem(
      label: 'Classification clinique',
      detail: d['predictedClass']?.toString() ?? '—',
      passed: !a.isInconclusive,
    ),
    VerificationItem(
      label: 'Confiance modèle',
      detail: '${(a.riskScore.confidence * 100).toStringAsFixed(0)} %',
      passed: a.riskScore.confidence >= 0.35,
      required: false,
    ),
  ];
}

/// User-facing alert cards for results screen.
class ResultAlertItem {
  final String title;
  final String message;
  final bool isWarning;

  const ResultAlertItem({
    required this.title,
    required this.message,
    this.isWarning = false,
  });
}

List<ResultAlertItem> buildResultAlerts(Assessment a) {
  final alerts = <ResultAlertItem>[];

  if (a.isInconclusive) {
    alerts.add(
      ResultAlertItem(
        title: 'Analyse non concluante',
        message: a.validationMessage ??
            'L’enregistrement ne permet pas une classification Control / MCI / AD.',
        isWarning: true,
      ),
    );
    final code = a.analysisDetails['validationCode']?.toString();
    if (code == 'too_short') {
      alerts.add(
        const ResultAlertItem(
          title: 'Durée insuffisante',
          message: 'Réenregistrez au moins 30 secondes en décrivant l’image.',
          isWarning: true,
        ),
      );
    }
    if (code == 'no_image_description' || code == 'insufficient_speech') {
      alerts.add(
        const ResultAlertItem(
          title: 'Description incomplète',
          message:
              'Décrivez les personnes, actions et objets visibles (garçon, femme, '
              'cookies, évier, eau, etc.).',
          isWarning: true,
        ),
      );
    }
    if (code == 'mostly_noise_or_silence') {
      alerts.add(
        const ResultAlertItem(
          title: 'Peu de parole détectée',
          message: 'Parlez plus fort et plus longtemps près du micro.',
          isWarning: true,
        ),
      );
    }
  } else {
    final cls = a.analysisDetails['predictedClass']?.toString() ?? '';
    if (cls == 'AD' || cls == 'MCI') {
      alerts.add(
        ResultAlertItem(
          title: 'Suivi recommandé',
          message: 'Le modèle suggère $cls — consultez un professionnel de santé.',
          isWarning: true,
        ),
      );
    } else {
      alerts.add(
        const ResultAlertItem(
          title: 'Profil vocal',
          message: 'Résultat dans la plage attendue pour un profil de contrôle.',
          isWarning: false,
        ),
      );
    }
  }

  alerts.add(
    const ResultAlertItem(
      title: 'Avertissement médical',
      message:
          'Ce test ne remplace pas un diagnostic médical. Les résultats sont indicatifs.',
      isWarning: true,
    ),
  );

  return alerts;
}
