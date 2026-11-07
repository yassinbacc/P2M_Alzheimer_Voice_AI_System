import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/assessment.dart';
import '../models/patient.dart';
import '../services/assessment_repository.dart';
import '../utils/assessment_verification.dart';
import '../widgets/app_alerts.dart';
import '../widgets/verification_checklist.dart';

class AssessmentResultsScreen extends StatefulWidget {
  final Assessment assessment;
  final Patient patient;

  const AssessmentResultsScreen({
    Key? key,
    required this.assessment,
    required this.patient,
  }) : super(key: key);

  @override
  State<AssessmentResultsScreen> createState() => _AssessmentResultsScreenState();
}

class _AssessmentResultsScreenState extends State<AssessmentResultsScreen>
    with SingleTickerProviderStateMixin {
  final AssessmentRepository _repo = AssessmentRepository();
  late TabController _tabs;

  bool _saved = false;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 4, vsync: this);
    if (widget.assessment.isInconclusive) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        showAppAlert(
          context,
          title: 'Analyse non concluante',
          message: widget.assessment.validationMessage ??
              'L’enregistrement n’a pas passé les contrôles qualité.',
          type: AppAlertType.warning,
        );
      });
    }
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  double get riskScore => widget.assessment.riskScore.score;
  double get confidence => widget.assessment.riskScore.confidence;
  String get riskLevel => widget.assessment.riskScore.riskLevel;
  String get predictedClass =>
      widget.assessment.analysisDetails['predictedClass']?.toString() ?? '—';
  bool get _inconclusive => widget.assessment.isInconclusive;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Résultats du test', style: TextStyle(fontWeight: FontWeight.bold)),
        bottom: TabBar(
          controller: _tabs,
          isScrollable: true,
          tabs: const [
            Tab(text: 'Résumé'),
            Tab(text: 'Vérification'),
            Tab(text: 'Alertes'),
            Tab(text: 'Détails'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabs,
        children: [
          _buildSummaryTab(),
          _buildVerificationTab(),
          _buildAlertsTab(),
          _buildDetailsTab(),
        ],
      ),
    );
  }

  Widget _buildSummaryTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeroCard(),
          const SizedBox(height: 20),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: _getRiskColor(riskScore).withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: _getRiskColor(riskScore)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Résumé', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Text(
                  _inconclusive
                      ? (widget.assessment.validationMessage ??
                          'Enregistrement non exploitable.')
                      : _getRiskDescription(riskScore),
                  style: const TextStyle(fontSize: 14, height: 1.6),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: () => Navigator.popUntil(context, (r) => r.isFirst),
                  icon: const Icon(Icons.home),
                  label: const Text('Accueil'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: (_saved || _saving) ? null : () => _saveLocally(),
                  icon: Icon(_saved ? Icons.check : Icons.save),
                  label: Text(_saved ? 'Enregistré' : 'Enregistrer'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    foregroundColor: Colors.white,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildVerificationTab() {
    final items = buildAssessmentVerification(widget.assessment);
    final allOk = items.where((i) => i.required).every((i) => i.passed);

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        AppAlertBanner(
          title: allOk ? 'Contrôles serveur OK' : 'Contrôles serveur échoués',
          subtitle: widget.assessment.analysisDetails['validationCode']?.toString() ?? '',
          type: allOk ? AppAlertType.success : AppAlertType.warning,
        ),
        VerificationChecklist(
          header: 'Validation audio & transcription',
          items: items,
          footer: 'Ces critères sont appliqués par le serveur avant classification.',
        ),
      ],
    );
  }

  Widget _buildAlertsTab() {
    final alerts = buildResultAlerts(widget.assessment);

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        const Text(
          'Messages et recommandations',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        ...alerts.map(
          (a) => AppAlertBanner(
            title: a.title,
            subtitle: a.message,
            type: a.isWarning ? AppAlertType.warning : AppAlertType.info,
          ),
        ),
        if (_inconclusive) ...[
          const SizedBox(height: 8),
          FilledButton.icon(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(Icons.mic),
            label: const Text('Réenregistrer'),
          ),
        ],
      ],
    );
  }

  Widget _buildDetailsTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Indicateurs vocaux',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          ...widget.assessment.riskScore.biomarkers.map(
            (b) => _buildBiomarkerItem(b, _getRiskColor(riskScore)),
          ),
          const SizedBox(height: 20),
          _buildAnalysisCard(
            'Transcription',
            widget.assessment.transcription.isEmpty
                ? 'Aucune transcription.'
                : widget.assessment.transcription,
          ),
          const SizedBox(height: 12),
          if (!_inconclusive)
            _buildAnalysisCard(
              'Probabilités',
              'AD: ${(_prob('prob_AD') * 100).toStringAsFixed(1)}%\n'
                  'Control: ${(_prob('prob_Control') * 100).toStringAsFixed(1)}%\n'
                  'MCI: ${(_prob('prob_MCI') * 100).toStringAsFixed(1)}%\n'
                  'Preset: ${widget.assessment.analysisDetails['preset'] ?? 'balanced'}',
            ),
          if (widget.assessment.analysisDetails['processingTimeSec'] != null)
            Padding(
              padding: const EdgeInsets.only(top: 12),
              child: _buildAnalysisCard(
                'Technique',
                'Temps de traitement : ${widget.assessment.analysisDetails['processingTimeSec']} s',
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildHeroCard() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: _inconclusive
              ? [Colors.orange.shade400, Colors.deepOrange.shade600]
              : [Colors.blue.shade400, Colors.blue.shade600],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Text(
            _inconclusive ? 'Non concluant' : 'Évaluation du risque',
            style: const TextStyle(color: Colors.white, fontSize: 16),
          ),
          const SizedBox(height: 8),
          Text(
            _inconclusive ? 'Non_concluant' : 'Classe: $predictedClass',
            style: const TextStyle(color: Colors.white70, fontSize: 14),
          ),
          const SizedBox(height: 16),
          if (_inconclusive)
            const Icon(Icons.warning_amber_rounded, color: Colors.white, size: 72)
          else
            Stack(
              alignment: Alignment.center,
              children: [
                SizedBox(
                  width: 120,
                  height: 120,
                  child: CircularProgressIndicator(
                    value: riskScore.clamp(0.0, 1.0),
                    strokeWidth: 8,
                    backgroundColor: Colors.white.withValues(alpha: 0.3),
                    valueColor: AlwaysStoppedAnimation<Color>(_getRiskColor(riskScore)),
                  ),
                ),
                Column(
                  children: [
                    Text(
                      '${(riskScore * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(riskLevel, style: const TextStyle(color: Colors.white, fontSize: 14)),
                  ],
                ),
              ],
            ),
          const SizedBox(height: 12),
          Text(
            'Confiance: ${(confidence * 100).toStringAsFixed(0)}%',
            style: const TextStyle(color: Colors.white, fontSize: 14),
          ),
        ],
      ),
    );
  }

  double _prob(String key) {
    final v = widget.assessment.analysisDetails[key];
    if (v is num) return v.toDouble();
    return 0;
  }

  Future<void> _saveLocally({bool silent = false}) async {
    if (_saving) return;
    if (mounted) setState(() => _saving = true);
    try {
      await _repo.save(widget.patient, widget.assessment);
      if (mounted) setState(() => _saved = true);
      if (!silent && mounted) {
        showAppSnack(
          context,
          kIsWeb ? 'Enregistré dans le navigateur.' : 'Enregistré localement.',
          type: AppAlertType.success,
        );
      }
    } catch (e) {
      if (!silent && mounted) {
        showAppAlert(context, title: 'Erreur', message: '$e', type: AppAlertType.error);
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Widget _buildBiomarkerItem(String description, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.5)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 8,
            height: 8,
            margin: const EdgeInsets.only(top: 6),
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 12),
          Expanded(child: Text(description, style: const TextStyle(fontSize: 13))),
        ],
      ),
    );
  }

  Widget _buildAnalysisCard(String title, String content) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13)),
          const SizedBox(height: 8),
          Text(content, style: TextStyle(fontSize: 12, color: Colors.grey.shade700, height: 1.5)),
        ],
      ),
    );
  }

  Color _getRiskColor(double score) {
    if (score < 0.35) return Colors.green;
    if (score < 0.65) return Colors.orange;
    return Colors.red;
  }

  String _getRiskDescription(double score) {
    if (score < 0.35) {
      return 'Risque faible — profil vocal proche des patterns de contrôle.';
    } else if (score < 0.65) {
      return 'Risque modéré — suivi clinique recommandé.';
    } else {
      return 'Risque élevé — consultez un professionnel de santé.';
    }
  }
}
