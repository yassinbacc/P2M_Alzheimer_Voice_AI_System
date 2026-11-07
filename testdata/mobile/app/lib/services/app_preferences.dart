import 'package:shared_preferences/shared_preferences.dart';

/// Lightweight session + inference settings (no auth server).
class AppPreferences {
  static const _keyPatientJson = 'current_patient_json';
  static const _keyPreset = 'inference_preset';

  static const presetChoices = ['balanced', 'mci_priority', 'control_priority'];

  Future<void> savePatientJson(String json) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyPatientJson, json);
  }

  Future<String?> loadPatientJson() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyPatientJson);
  }

  Future<void> clearSession() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_keyPatientJson);
  }

  Future<void> setPreset(String preset) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyPreset, preset);
  }

  Future<String> getPreset() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyPreset) ?? 'balanced';
  }
}
