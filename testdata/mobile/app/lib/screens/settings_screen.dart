import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../services/app_preferences.dart';
import '../services/session_service.dart';
import '../utils/api_config.dart';

class SettingsScreen extends StatefulWidget {
  final VoidCallback onLogout;

  const SettingsScreen({Key? key, required this.onLogout}) : super(key: key);

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final AppPreferences _prefs = AppPreferences();
  final SessionService _session = SessionService();
  final ApiService _api = ApiService();

  String _preset = 'balanced';
  bool _apiOk = false;
  String _apiStatus = 'Non vérifié';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final preset = await _prefs.getPreset();
    if (mounted) setState(() => _preset = preset);
    await _checkApi();
  }

  Future<void> _checkApi() async {
    setState(() => _apiStatus = 'Vérification…');
    final ok = await _api.healthCheck();
    if (!mounted) return;
    setState(() {
      _apiOk = ok;
      _apiStatus = ok ? 'Serveur connecté' : 'Serveur inaccessible';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Paramètres', style: TextStyle(fontWeight: FontWeight.bold)),
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text('Analyse',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Card(
            child: ListTile(
              leading: const Icon(Icons.tune, color: Color(0xFF2F5BFF)),
              title: const Text('Préréglage du modèle'),
              subtitle: Text(_presetLabel(_preset)),
              trailing: DropdownButton<String>(
                value: _preset,
                items: AppPreferences.presetChoices
                    .map((p) => DropdownMenuItem(value: p, child: Text(_presetLabel(p))))
                    .toList(),
                onChanged: (v) async {
                  if (v == null) return;
                  await _prefs.setPreset(v);
                  setState(() => _preset = v);
                },
              ),
            ),
          ),
          const SizedBox(height: 24),
          const Text('Serveur API',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Card(
            child: ListTile(
              leading: Icon(
                _apiOk ? Icons.cloud_done : Icons.cloud_off,
                color: _apiOk ? Colors.green : Colors.red,
              ),
              title: const Text('Connexion backend'),
              subtitle: Text('${ApiConfig.baseUrl}\n$_apiStatus'),
              isThreeLine: true,
              trailing: IconButton(
                icon: const Icon(Icons.refresh),
                onPressed: _checkApi,
              ),
            ),
          ),
          const SizedBox(height: 24),
          const Text('Compte',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Card(
            child: ListTile(
              leading: const Icon(Icons.person, color: Color(0xFF2F5BFF)),
              title: const Text('Profil patient'),
              onTap: () => Navigator.pushNamed(context, '/patient-profile'),
            ),
          ),
          const SizedBox(height: 24),
          const Text('Données',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.red)),
          const SizedBox(height: 8),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () => _confirmClear(context),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.red,
                foregroundColor: Colors.white,
              ),
              child: const Text('Effacer toutes les données locales'),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              onPressed: widget.onLogout,
              child: const Text('Déconnexion'),
            ),
          ),
          const SizedBox(height: 32),
          const Center(
            child: Text(
              'AlzheimerAI v1.1.0\nUsage recherche / éducatif uniquement.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.black54, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  String _presetLabel(String p) => switch (p) {
        'mci_priority' => 'Priorité MCI',
        'control_priority' => 'Priorité Contrôle',
        _ => 'Équilibré',
      };

  Future<void> _confirmClear(BuildContext context) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Effacer les données ?'),
        content: const Text(
          'Supprime les patients et résultats enregistrés sur cet appareil.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Annuler')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Effacer', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
    if (ok == true) {
      await _session.clearAll();
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Données locales effacées')),
        );
        widget.onLogout();
      }
    }
  }
}
