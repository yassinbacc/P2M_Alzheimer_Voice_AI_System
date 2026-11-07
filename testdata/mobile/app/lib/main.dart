import 'package:flutter/material.dart';

import 'models/patient.dart';
import 'screens/audio_recording_screen.dart';
import 'screens/home_screen.dart';
import 'screens/login_screen.dart';
import 'screens/patient_history_screen.dart';
import 'screens/patient_profile_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/assessment_results_screen.dart';
import 'models/assessment.dart';
import 'services/session_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AlzheimerDetectionApp());
}

class AlzheimerDetectionApp extends StatefulWidget {
  const AlzheimerDetectionApp({Key? key}) : super(key: key);

  @override
  State<AlzheimerDetectionApp> createState() => _AlzheimerDetectionAppState();
}

class _AlzheimerDetectionAppState extends State<AlzheimerDetectionApp> {
  final GlobalKey<NavigatorState> _navigatorKey = GlobalKey<NavigatorState>();
  final SessionService _session = SessionService();
  Patient? _currentPatient;
  bool _bootstrapping = true;

  @override
  void initState() {
    super.initState();
    _restoreSession();
  }

  Future<void> _restoreSession() async {
    final patient = await _session.loadPatient();
    if (!mounted) return;
    setState(() {
      _currentPatient = patient;
      _bootstrapping = false;
    });
  }

  Future<void> _handleLogin(Patient patient) async {
    try {
      await _session.savePatient(patient);
    } catch (e) {
      debugPrint('Login save error: $e');
    }
    if (!mounted) return;
    setState(() => _currentPatient = patient);
  }

  Future<void> _handleLogout() async {
    await _session.clearAll();
    if (!mounted) return;
    setState(() => _currentPatient = null);
  }

  @override
  Widget build(BuildContext context) {
    if (_bootstrapping) {
      return const MaterialApp(
        home: Scaffold(
          body: Center(child: CircularProgressIndicator()),
        ),
      );
    }

    return MaterialApp(
      navigatorKey: _navigatorKey,
      title: 'AlzheimerAI Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primaryColor: const Color(0xFF2F5BFF),
        scaffoldBackgroundColor: const Color(0xFFF5F8FF),
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2F5BFF),
          brightness: Brightness.light,
        ),
        appBarTheme: const AppBarTheme(
          centerTitle: true,
          elevation: 0,
          color: Colors.white,
          foregroundColor: Colors.black87,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF2F5BFF),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          contentPadding: const EdgeInsets.all(18),
        ),
        cardTheme: const CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(20)),
          ),
        ),
      ),
      home: _currentPatient != null
          ? HomeScreen(patient: _currentPatient!)
          : LoginScreen(onLogin: _handleLogin),
      routes: {
        '/home': (context) => HomeScreen(patient: _currentPatient!),
        '/patient-profile': (context) {
          final patient = ModalRoute.of(context)?.settings.arguments as Patient? ??
              _currentPatient!;
          return PatientProfileScreen(patient: patient);
        },
        '/audio-recording': (context) {
          final patient = ModalRoute.of(context)?.settings.arguments as Patient? ??
              _currentPatient!;
          return AudioRecordingScreen(patient: patient);
        },
        '/patient-history': (context) {
          final patient = ModalRoute.of(context)?.settings.arguments as Patient? ??
              _currentPatient!;
          return PatientHistoryScreen(patient: patient);
        },
        '/settings': (context) => SettingsScreen(onLogout: _handleLogout),
        '/assessment-results': (context) {
          final args = ModalRoute.of(context)!.settings.arguments as Map<String, dynamic>;
          return AssessmentResultsScreen(
            assessment: args['assessment'] as Assessment,
            patient: args['patient'] as Patient,
          );
        },
      },
    );
  }
}
