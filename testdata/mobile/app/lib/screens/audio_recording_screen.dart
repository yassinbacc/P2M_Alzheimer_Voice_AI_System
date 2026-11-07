import 'dart:async';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

import '../models/patient.dart';
import '../services/api_service.dart';
import '../services/app_preferences.dart';
import '../services/audio_service.dart';
import '../utils/audio_validation_constants.dart';
import '../utils/recording_path.dart';
import '../utils/recording_verification.dart';
import '../utils/web_blob_duration.dart';
import '../utils/wav_duration.dart';
import '../widgets/app_alerts.dart';
import '../widgets/verification_checklist.dart';
import 'assessment_results_screen.dart';

class AudioRecordingScreen extends StatefulWidget {
  final Patient patient;

  const AudioRecordingScreen({Key? key, required this.patient}) : super(key: key);

  @override
  State<AudioRecordingScreen> createState() => _AudioRecordingScreenState();
}

class _AudioRecordingScreenState extends State<AudioRecordingScreen>
    with TickerProviderStateMixin {
  final AudioService _audio = AudioService();
  final ApiService _api = ApiService();
  final AppPreferences _prefs = AppPreferences();

  late TabController _tabs;
  late Ticker _clockTicker;

  final ValueNotifier<String> _clock = ValueNotifier('00:00');
  final Stopwatch _stopwatch = Stopwatch();

  bool _recording = false;
  bool _busy = false;
  bool _micOk = true;
  bool _hasClip = false;
  /// Duration from WAV PCM bytes (same basis as server analysis).
  double _clipDurationSec = 0;

  String? _path;
  Uint8List? _bytes;
  String? _filename;

  Timer? _analyzeTimer;
  int _analyzeSec = 0;

  static const _blue = Color(0xFF2F5BFF);
  static const _red = Color(0xFFEB5757);
  static const _green = Color(0xFF27AE60);

  int get _minSec => AudioValidationConstants.minDurationSeconds;

  List<VerificationItem> get _checks => buildRecordingVerification(
        micOk: _micOk,
        hasClip: _hasClip,
        durationSec: _clipDurationSec,
        fileBytes: _bytes?.length,
        isRecording: _recording,
      );

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 2, vsync: this);
    _clockTicker = createTicker(_onClockTick);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _checkMic();
    });
  }

  void _onClockTick(Duration _) {
    if (!_recording) return;
    final sec = kIsWeb
        ? _audio.webCaptureSeconds.value.floor()
        : (_stopwatch.isRunning ? _stopwatch.elapsed.inSeconds : 0);
    final text = _fmt(sec);
    if (_clock.value != text) _clock.value = text;
  }

  @override
  void dispose() {
    _tabs.dispose();
    _haltRecordingClock();
    _clockTicker.dispose();
    _analyzeTimer?.cancel();
    _clock.dispose();
    _audio.dispose();
    super.dispose();
  }

  /// Starts only when the microphone is actually recording.
  void _beginRecordingClock() {
    if (!kIsWeb) {
      _stopwatch
        ..reset()
        ..start();
    } else {
      _audio.webCaptureSeconds.value = 0;
    }
    _clock.value = '00:00';
    if (!_clockTicker.isActive) _clockTicker.start();
  }

  void _haltRecordingClock() {
    if (_stopwatch.isRunning) _stopwatch.stop();
    if (_clockTicker.isActive) _clockTicker.stop();
  }

  static String _fmt(int sec) {
    final m = sec ~/ 60;
    final s = sec % 60;
    return '${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
  }

  static String _fmtDurationLabel(double sec) =>
      '${sec.toStringAsFixed(1)} s (${_fmt(sec.floor())})';

  void _applyCapture(RecordingCapture cap) {
    _bytes = cap.bytes;
    _path = cap.path;
    _filename = cap.displayName;
    _clipDurationSec = (cap.durationSec * 10).roundToDouble() / 10;
    setState(() {
      _recording = false;
      _hasClip = true;
      _clock.value = _fmt(cap.durationSec.floor());
    });

    if (_clipDurationSec < _minSec) {
      showAppAlert(
        context,
        title: 'Durée courte',
        message: 'Audio enregistré : ${_fmtDurationLabel(_clipDurationSec)} — '
            'minimum $_minSec s pour analyser. Consultez l’onglet Vérification.',
        type: AppAlertType.warning,
      );
      _tabs.animateTo(1);
    } else {
      showAppSnack(
        context,
        'Audio prêt (${_fmtDurationLabel(_clipDurationSec)}).',
        type: AppAlertType.success,
      );
    }
  }

  Future<void> _checkMic() async {
    final ok = await _audio.requestPermissions();
    if (!mounted) return;
    setState(() => _micOk = ok);
    if (!ok) {
      showAppAlert(
        context,
        title: 'Microphone requis',
        message: 'Autorisez le micro (cadenas Chrome) puis rechargez la page (F5).',
        type: AppAlertType.warning,
      );
    }
  }

  Future<void> _start() async {
    if (_busy || _recording) return;
    if (!_micOk) {
      await _checkMic();
      if (!_micOk) return;
    }

    setState(() {
      _busy = true;
      _hasClip = false;
      _path = null;
      _bytes = null;
      _filename = null;
      _clipDurationSec = 0;
      _recording = false;
    });
    _haltRecordingClock();
    _clock.value = '00:00';

    if (_audio.isRecording) await _audio.cancelRecording();

    final path = await _audio.startRecording();
    if (!mounted) return;

    if (path == null) {
      setState(() => _busy = false);
      showAppAlert(
        context,
        title: 'Erreur micro',
        message: 'Impossible de démarrer l’enregistrement.',
        type: AppAlertType.error,
      );
      return;
    }

    _beginRecordingClock();
    setState(() {
      _busy = false;
      _recording = true;
      _path = path;
      _filename = 'recording.wav';
    });
    showAppSnack(context, 'Enregistrement démarré — parlez en décrivant l’image.');
  }

  Future<void> _stop() async {
    if (_busy || !_recording) return;
    setState(() => _busy = true);

    final cap = await _audio.stopRecording();
    _haltRecordingClock();

    if (!mounted) return;
    setState(() => _busy = false);

    if (cap == null || !cap.hasAudio) {
      setState(() {
        _recording = false;
        _hasClip = false;
        _clipDurationSec = 0;
        _clock.value = '00:00';
      });
      await _audio.cancelRecording();
      showAppAlert(
        context,
        title: 'Enregistrement vide',
        message: 'Aucun son capturé. Vérifiez le micro ou importez un fichier WAV.',
        type: AppAlertType.error,
      );
      return;
    }

    _applyCapture(cap);
  }

  Future<void> _reset() async {
    if (_busy) return;
    if (_recording) {
      _haltRecordingClock();
      await _audio.cancelRecording();
    }
    setState(() {
      _recording = false;
      _hasClip = false;
      _busy = false;
      _clipDurationSec = 0;
      _path = null;
      _bytes = null;
      _filename = null;
      _clock.value = '00:00';
    });
    showAppSnack(context, 'Enregistrement effacé.');
  }

  Future<void> _import() async {
    if (_busy || _recording) return;
    final r = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav', 'mp3', 'm4a', 'ogg', 'flac'],
      withData: true,
    );
    if (r == null || r.files.isEmpty) return;

    final f = r.files.single;
    var data = f.bytes;
    if (data == null && f.path != null && !kIsWeb) {
      data = await readRecordingBytes(f.path!);
    }
    if (data == null || data.isEmpty) {
      showAppAlert(context, title: 'Import', message: 'Fichier illisible.', type: AppAlertType.error);
      return;
    }

    final name = f.name.toLowerCase();
    final isWav = name.endsWith('.wav');
    Uint8List payload = data;
    double sec = 0;

    if (isWav) {
      payload = repairWavHeader(data);
      sec = wavDurationSeconds(payload) ?? 0;
    } else if (kIsWeb) {
      sec = await audioBlobDurationSeconds(
            data,
            mimeType: name.endsWith('.webm') ? 'audio/webm' : 'audio/mp4',
          ) ??
          0;
    }

    if (sec <= 0) {
      showAppAlert(
        context,
        title: 'Import',
        message: 'Impossible de lire la durée du fichier audio.',
        type: AppAlertType.error,
      );
      return;
    }
    _clipDurationSec = (sec * 10).roundToDouble() / 10;
    setState(() {
      _bytes = payload;
      _filename = f.name;
      _path = kIsWeb ? f.name : f.path;
      _hasClip = true;
      _recording = false;
      _clock.value = _fmt(sec.floor());
    });
    _tabs.animateTo(1);
    showAppSnack(context, 'Fichier importé : ${f.name}', type: AppAlertType.success);
  }

  Future<void> _analyze() async {
    if (_busy || _recording || !_hasClip) return;

    final checks = _checks;
    if (!recordingReadyForAnalysis(checks)) {
      _tabs.animateTo(1);
      showAppAlert(
        context,
        title: 'Vérification incomplète',
        message: 'Corrigez les points en rouge dans l’onglet Vérification avant d’analyser.',
        type: AppAlertType.warning,
      );
      return;
    }

    setState(() => _busy = true);
    _analyzeSec = 0;
    _analyzeTimer?.cancel();
    _analyzeTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() => _analyzeSec++);
    });

    try {
      final preset = await _prefs.getPreset();
      final assessment = await _api.analyzeAudio(
        widget.patient.id,
        _path ?? _filename ?? 'audio.wav',
        preset: preset,
        audioBytes: _bytes,
        audioFilename: _filename ?? 'recording.wav',
      );
      if (!mounted) return;
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => AssessmentResultsScreen(
            assessment: assessment,
            patient: widget.patient,
          ),
        ),
      );
    } catch (e) {
      if (mounted) {
        showAppAlert(
          context,
          title: 'Analyse impossible',
          message: 'Une erreur est survenue pendant l’analyse. Réessayez.',
          type: AppAlertType.error,
        );
      }
    } finally {
      _analyzeTimer?.cancel();
      if (mounted) setState(() => _busy = false);
    }
  }

  List<Widget> _activeAlerts() {
    final alerts = <Widget>[];
    if (!_micOk) {
      alerts.add(const AppAlertBanner(
        title: 'Micro non autorisé',
        subtitle: 'Autorisez l’accès au microphone.',
        type: AppAlertType.error,
      ));
    }
    if (_recording) {
      alerts.add(ValueListenableBuilder<String>(
        valueListenable: _clock,
        builder: (_, time, __) => AppAlertBanner(
          title: 'Enregistrement en cours — $time',
          subtitle: kIsWeb
              ? 'Gardez cet onglet Chrome au premier plan jusqu’à « Arrêter » (min. $_minSec s).'
              : 'Appuyez sur Arrêter quand vous avez fini (min. $_minSec s).',
          type: AppAlertType.info,
        ),
      ));
    }
    if (_hasClip && _clipDurationSec < _minSec) {
      alerts.add(AppAlertBanner(
        title: 'Durée insuffisante',
        subtitle: 'Minimum $_minSec s (audio : ${_clipDurationSec.toStringAsFixed(1)} s).',
        type: AppAlertType.warning,
      ));
    }
    return alerts;
  }

  Widget _buildRecordTab() {
    final canAnalyze = _hasClip && !_recording && !_busy && recordingReadyForAnalysis(_checks);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ..._activeAlerts(),
        Expanded(
          child: DecoratedBox(
            decoration: BoxDecoration(
              color: const Color(0xFFE8F0FF),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: _blue),
            ),
            child: Image.asset(
              'assets/images/coockie_theft.png',
              fit: BoxFit.contain,
              errorBuilder: (_, __, ___) =>
                  const Icon(Icons.image_not_supported_outlined, size: 64, color: _blue),
            ),
          ),
        ),
        const SizedBox(height: 20),
        ValueListenableBuilder<String>(
          valueListenable: _clock,
          builder: (_, time, __) => Text(
            time,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 48,
              fontWeight: FontWeight.w300,
              color: _recording ? _red : Colors.black87,
              letterSpacing: 3,
            ),
          ),
        ),
        if (_filename != null && _hasClip && !_recording)
          Text(_filename!, textAlign: TextAlign.center, style: const TextStyle(fontSize: 12)),
        const SizedBox(height: 20),
        if (_busy && !_recording && !_hasClip)
          const Center(child: CircularProgressIndicator())
        else
          FilledButton.icon(
            onPressed: (_busy && !_recording) || !_micOk
                ? null
                : _recording
                    ? _stop
                    : _start,
            icon: Icon(_recording ? Icons.stop : Icons.mic),
            label: Text(_recording ? 'Arrêter' : 'Démarrer'),
            style: FilledButton.styleFrom(
              backgroundColor: _recording ? _red : _blue,
              minimumSize: const Size(double.infinity, 52),
            ),
          ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: OutlinedButton(onPressed: (_busy || _recording) ? null : _reset, child: const Text('Effacer')),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: (_busy || _recording) ? null : _import,
                icon: const Icon(Icons.upload_file, size: 18),
                label: const Text('Importer'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        FilledButton(
          onPressed: _busy && _hasClip && !_recording
              ? null
              : canAnalyze
                  ? _analyze
                  : () {
                      _tabs.animateTo(1);
                      showAppAlert(
                        context,
                        title: 'Vérification requise',
                        message: 'Complétez la checklist dans l’onglet Vérification.',
                        type: AppAlertType.warning,
                      );
                    },
          style: FilledButton.styleFrom(
            backgroundColor: canAnalyze ? _green : Colors.grey,
            minimumSize: const Size(double.infinity, 48),
          ),
          child: Text(
            _busy && _hasClip && !_recording
                ? 'Analyse… ${_fmt(_analyzeSec)}'
                : canAnalyze
                    ? 'Analyser'
                    : 'Analyser (vérifier d’abord)',
          ),
        ),
      ],
    );
  }

  Widget _buildVerifyTab() {
    final checks = _checks;
    final ready = recordingReadyForAnalysis(checks);

    return ListView(
      padding: const EdgeInsets.only(top: 8, bottom: 24),
      children: [
        if (ready)
          const AppAlertBanner(
            title: 'Tout est prêt',
            subtitle: 'Vous pouvez lancer l’analyse.',
            type: AppAlertType.success,
          )
        else
          const AppAlertBanner(
            title: 'Actions requises',
            subtitle: 'Corrigez les éléments en rouge ci-dessous.',
            type: AppAlertType.warning,
          ),
        VerificationChecklist(
          header: 'Contrôles avant analyse',
          items: checks,
          footer:
              'Durée minimale : $_minSec s. Décrivez l’image Cookie Theft '
              '(personnes, actions, cuisine, eau, etc.).',
        ),
        const SizedBox(height: 12),
        if (!_micOk) ...[
          const SizedBox(height: 8),
          TextButton.icon(
            onPressed: _checkMic,
            icon: const Icon(Icons.mic_external_on),
            label: const Text('Vérifier le microphone'),
          ),
        ],
        const SizedBox(height: 12),
        FilledButton.icon(
          onPressed: ready && !_busy && !_recording ? _analyze : null,
          icon: const Icon(Icons.science_outlined),
          label: const Text('Lancer l’analyse'),
          style: FilledButton.styleFrom(
            backgroundColor: _green,
            minimumSize: const Size(double.infinity, 48),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: !_recording,
      onPopInvokedWithResult: (pop, _) async {
        if (pop || !_recording) return;
        final leave = await showAppConfirm(
          context,
          title: 'Quitter ?',
          message: 'L’enregistrement en cours sera perdu.',
        );
        if (leave == true && mounted) {
          await _audio.cancelRecording();
          if (mounted) Navigator.pop(context);
        }
      },
      child: Scaffold(
        appBar: AppBar(
          title: ValueListenableBuilder<String>(
            valueListenable: _clock,
            builder: (_, time, __) {
              if (_recording) {
                return Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 10,
                      height: 10,
                      margin: const EdgeInsets.only(right: 8),
                      decoration: const BoxDecoration(
                        color: _red,
                        shape: BoxShape.circle,
                      ),
                    ),
                    Text(
                      time,
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontFeatures: [FontFeature.tabularFigures()],
                      ),
                    ),
                  ],
                );
              }
              return Text('Enregistrement — ${widget.patient.firstName}');
            },
          ),
          bottom: TabBar(
            controller: _tabs,
            tabs: const [
              Tab(icon: Icon(Icons.mic), text: 'Enregistrer'),
              Tab(icon: Icon(Icons.fact_check), text: 'Vérification'),
            ],
          ),
        ),
        body: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: TabBarView(
            controller: _tabs,
            children: [
              _KeepAliveTab(child: _buildRecordTab()),
              _KeepAliveTab(child: _buildVerifyTab()),
            ],
          ),
        ),
      ),
    );
  }
}

/// Keeps tab content alive so switching tabs does not disrupt recording UI.
class _KeepAliveTab extends StatefulWidget {
  final Widget child;
  const _KeepAliveTab({required this.child});

  @override
  State<_KeepAliveTab> createState() => _KeepAliveTabState();
}

class _KeepAliveTabState extends State<_KeepAliveTab> with AutomaticKeepAliveClientMixin {
  @override
  bool get wantKeepAlive => true;

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return widget.child;
  }
}
