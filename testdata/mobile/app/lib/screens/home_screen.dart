import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/assessment.dart';
import '../models/patient.dart';
import '../services/assessment_repository.dart';

class HomeScreen extends StatefulWidget {
  final Patient patient;

  const HomeScreen({Key? key, required this.patient}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
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
      debugPrint('Home load assessments: $e');
    }
    if (mounted) {
      setState(() {
        _assessments = list;
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final count = _assessments.length;
    final last = _assessments.isNotEmpty ? _assessments.first : null;
    final lastRisk = last?.riskScore.riskLevel ?? '—';
    final lastDate = last != null
        ? DateFormat('dd/MM').format(last.recordedAt)
        : '—';

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: const Text('AlzheimerAI', style: TextStyle(fontWeight: FontWeight.bold)),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(widget.patient.firstName,
                    style: const TextStyle(fontWeight: FontWeight.bold)),
                Text('ID: ${widget.patient.id}',
                    style: const TextStyle(fontSize: 12, color: Colors.black54)),
              ],
            ),
          ),
        ],
        elevation: 0,
      ),
      body: RefreshIndicator(
        onRefresh: _load,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text('Bienvenue,',
                  style: TextStyle(fontSize: 18, color: Colors.black54)),
              Text(widget.patient.firstName,
                  style: const TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              const Text('Que souhaitez-vous faire aujourd’hui ?',
                  style: TextStyle(fontSize: 16, color: Colors.black54)),
              const SizedBox(height: 24),
              Row(
                children: [
                  Expanded(
                    child: _buildLargeCard(
                      icon: Icons.mic,
                      title: 'Nouveau Test',
                      subtitle: 'Enregistrer ou importer un audio',
                      color: const Color(0xFFEEF3FF),
                      iconBackground: const Color(0xFFD9E4FF),
                      onTap: () async {
                        await Navigator.pushNamed(
                          context,
                          '/audio-recording',
                          arguments: widget.patient,
                        );
                        _load();
                      },
                    ),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: _buildLargeCard(
                      icon: Icons.history,
                      title: 'Mes Résultats',
                      subtitle: 'Historique de vos analyses',
                      color: const Color(0xFFF8F0FF),
                      iconBackground: const Color(0xFFEBD9FF),
                      onTap: () async {
                        await Navigator.pushNamed(
                          context,
                          '/patient-history',
                          arguments: widget.patient,
                        );
                        _load();
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 24),
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(24),
                  gradient: const LinearGradient(
                    colors: [Color(0xFF3D69FF), Color(0xFF8A4DFF)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  boxShadow: const [
                    BoxShadow(
                        color: Colors.black12, blurRadius: 24, offset: Offset(0, 12)),
                  ],
                ),
                child: const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Comment ça marche ?',
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 20,
                            fontWeight: FontWeight.bold)),
                    SizedBox(height: 16),
                    Text('1. Décrivez l’image Cookie Theft à voix haute',
                        style: TextStyle(color: Colors.white, fontSize: 15, height: 1.6)),
                    Text('2. Importez ou enregistrez l’audio (≥ 5 s)',
                        style: TextStyle(color: Colors.white, fontSize: 15, height: 1.6)),
                    Text('3. L’analyse prend quelques minutes (Whisper + IA)',
                        style: TextStyle(color: Colors.white, fontSize: 15, height: 1.6)),
                    Text('4. Consultez le risque et l’historique',
                        style: TextStyle(color: Colors.white, fontSize: 15, height: 1.6)),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              if (_loading)
                const Center(child: Padding(
                  padding: EdgeInsets.all(24),
                  child: CircularProgressIndicator(),
                ))
              else
                Row(
                  children: [
                    Expanded(
                        child: _buildStatusCard(
                            '$count', 'Tests effectués', Colors.blue)),
                    const SizedBox(width: 12),
                    Expanded(
                        child: _buildStatusCard(
                            lastRisk, 'Dernier résultat', Colors.green)),
                    const SizedBox(width: 12),
                    Expanded(
                        child: _buildStatusCard(
                            lastDate, 'Dernier test', Colors.black87)),
                  ],
                ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 0,
        onTap: (index) {
          switch (index) {
            case 0:
              break;
            case 1:
              Navigator.pushNamed(context, '/audio-recording',
                  arguments: widget.patient);
              break;
            case 2:
              Navigator.pushNamed(context, '/patient-history',
                  arguments: widget.patient);
              break;
            case 3:
              Navigator.pushNamed(context, '/settings');
              break;
          }
        },
        selectedItemColor: const Color(0xFF2F5BFF),
        unselectedItemColor: Colors.black54,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Accueil'),
          BottomNavigationBarItem(icon: Icon(Icons.mic), label: 'Test'),
          BottomNavigationBarItem(icon: Icon(Icons.history), label: 'Historique'),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: 'Paramètres'),
        ],
      ),
    );
  }

  Widget _buildLargeCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color color,
    required Color iconBackground,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(24),
          boxShadow: const [
            BoxShadow(color: Colors.black12, blurRadius: 18, offset: Offset(0, 10)),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 54,
              height: 54,
              decoration: BoxDecoration(
                color: iconBackground,
                borderRadius: BorderRadius.circular(18),
              ),
              child: Icon(icon, color: const Color(0xFF2F5BFF), size: 28),
            ),
            const SizedBox(height: 18),
            Text(title, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            Text(subtitle, style: const TextStyle(color: Colors.black54, height: 1.4)),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusCard(String value, String label, Color color) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: const [
          BoxShadow(color: Colors.black12, blurRadius: 20, offset: Offset(0, 10)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(value,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: color)),
          const SizedBox(height: 8),
          Text(label, style: const TextStyle(color: Colors.black54, fontSize: 12)),
        ],
      ),
    );
  }
}
