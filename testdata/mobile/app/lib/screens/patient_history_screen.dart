import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/assessment.dart';
import '../models/patient.dart';
import '../services/assessment_repository.dart';

class PatientHistoryScreen extends StatefulWidget {
  final Patient patient;

  const PatientHistoryScreen({Key? key, required this.patient}) : super(key: key);

  @override
  State<PatientHistoryScreen> createState() => _PatientHistoryScreenState();
}

class _PatientHistoryScreenState extends State<PatientHistoryScreen> {
  final AssessmentRepository _repo = AssessmentRepository();
  List<Assessment> _assessments = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    List<Assessment> list = [];
    try {
      list = await _repo.listForPatient(widget.patient.id);
    } catch (e) {
      debugPrint('History load: $e');
    }
    if (mounted) {
      setState(() {
        _assessments = list;
        _loading = false;
      });
    }
  }

  Color riskColor(double score) {
    if (score < 0.35) return Colors.green;
    if (score < 0.65) return Colors.orange;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    final last = _assessments.isNotEmpty ? _assessments.first : null;
    final chartItems = _assessments.take(6).toList().reversed.toList();

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: const Text('Historique des tests',
            style: TextStyle(fontWeight: FontWeight.bold)),
        elevation: 0,
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: SingleChildScrollView(
                physics: const AlwaysScrollableScrollPhysics(),
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                            child: _buildSummaryCard(
                                '${_assessments.length}', 'Tests', Colors.blue)),
                        const SizedBox(width: 12),
                        Expanded(
                            child: _buildSummaryCard(
                                last?.riskScore.riskLevel ?? '—',
                                'Dernier',
                                Colors.green)),
                        const SizedBox(width: 12),
                        Expanded(
                            child: _buildSummaryCard(
                                last != null
                                    ? DateFormat('dd/MM').format(last.recordedAt)
                                    : '—',
                                'Date',
                                Colors.black87)),
                      ],
                    ),
                    const SizedBox(height: 24),
                    const Text('Tendance du risque',
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 16),
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(24),
                        boxShadow: const [
                          BoxShadow(
                              color: Colors.black12,
                              blurRadius: 22,
                              offset: Offset(0, 12)),
                        ],
                      ),
                      child: chartItems.isEmpty
                          ? const SizedBox(
                              height: 120,
                              child: Center(
                                child: Text('Aucun test enregistré',
                                    style: TextStyle(color: Colors.black54)),
                              ),
                            )
                          : SizedBox(
                              height: 180,
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                                crossAxisAlignment: CrossAxisAlignment.end,
                                children: chartItems.map((a) {
                                  final score = a.riskScore.score.clamp(0.0, 1.0);
                                  return Column(
                                    mainAxisAlignment: MainAxisAlignment.end,
                                    children: [
                                      Container(
                                        width: 22,
                                        height: (score * 140).clamp(8.0, 140.0),
                                        decoration: BoxDecoration(
                                          color: riskColor(score),
                                          borderRadius: const BorderRadius.vertical(
                                              top: Radius.circular(8)),
                                        ),
                                      ),
                                      const SizedBox(height: 8),
                                      Text(
                                        DateFormat('d/M').format(a.recordedAt),
                                        style: const TextStyle(
                                            fontSize: 10, color: Colors.black54),
                                      ),
                                    ],
                                  );
                                }).toList(),
                              ),
                            ),
                    ),
                    const SizedBox(height: 24),
                    const Text('Tests précédents',
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 16),
                    if (_assessments.isEmpty)
                      const Center(
                        child: Padding(
                          padding: EdgeInsets.all(32),
                          child: Text(
                            'Aucun résultat. Lancez un premier test vocal.',
                            textAlign: TextAlign.center,
                            style: TextStyle(color: Colors.black54),
                          ),
                        ),
                      )
                    else
                      ..._assessments.map((a) {
                        final score = a.riskScore.score;
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: Card(
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(20)),
                            child: ListTile(
                              contentPadding: const EdgeInsets.all(18),
                              leading: Container(
                                width: 54,
                                height: 54,
                                decoration: BoxDecoration(
                                  color: riskColor(score).withOpacity(0.2),
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                child: Center(
                                  child: Text(
                                    '${(score * 100).toInt()}%',
                                    style: TextStyle(
                                        fontWeight: FontWeight.bold,
                                        color: riskColor(score)),
                                  ),
                                ),
                              ),
                              title: Text(a.formattedDate,
                                  style: const TextStyle(fontWeight: FontWeight.bold)),
                              subtitle: Text(
                                  '${a.riskScore.riskLevel} · ${a.formattedDuration}'),
                              trailing: const Icon(Icons.arrow_forward_ios, size: 18),
                              onTap: () {
                                Navigator.pushNamed(
                                  context,
                                  '/assessment-results',
                                  arguments: {
                                    'assessment': a,
                                    'patient': widget.patient,
                                  },
                                );
                              },
                            ),
                          ),
                        );
                      }),
                    const SizedBox(height: 24),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed: () {
                          Navigator.pushNamed(
                            context,
                            '/audio-recording',
                            arguments: widget.patient,
                          );
                        },
                        icon: const Icon(Icons.mic),
                        label: const Text('Nouveau test'),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 18),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16)),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
    );
  }

  Widget _buildSummaryCard(String value, String label, Color color) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: const [
          BoxShadow(color: Colors.black12, blurRadius: 18, offset: Offset(0, 10)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(value,
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: color)),
          const SizedBox(height: 10),
          Text(label, style: const TextStyle(color: Colors.black54, fontSize: 12)),
        ],
      ),
    );
  }
}
