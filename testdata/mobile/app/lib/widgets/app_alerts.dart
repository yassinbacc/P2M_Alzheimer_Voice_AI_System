import 'package:flutter/material.dart';

enum AppAlertType { info, success, warning, error }

/// Floating snackbar.
void showAppSnack(
  BuildContext context,
  String message, {
  AppAlertType type = AppAlertType.info,
}) {
  final color = switch (type) {
    AppAlertType.success => const Color(0xFF27AE60),
    AppAlertType.warning => const Color(0xFFF2994A),
    AppAlertType.error => const Color(0xFFEB5757),
    AppAlertType.info => null,
  };
  ScaffoldMessenger.of(context)
    ..hideCurrentSnackBar()
    ..showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: color,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 4),
      ),
    );
}

/// Modal dialog alert.
Future<void> showAppAlert(
  BuildContext context, {
  required String title,
  required String message,
  AppAlertType type = AppAlertType.info,
  String okLabel = 'OK',
}) {
  final icon = switch (type) {
    AppAlertType.success => Icons.check_circle_outline,
    AppAlertType.warning => Icons.warning_amber_rounded,
    AppAlertType.error => Icons.error_outline,
    AppAlertType.info => Icons.info_outline,
  };
  final color = switch (type) {
    AppAlertType.success => const Color(0xFF27AE60),
    AppAlertType.warning => const Color(0xFFF2994A),
    AppAlertType.error => const Color(0xFFEB5757),
    AppAlertType.info => const Color(0xFF2F5BFF),
  };

  return showDialog<void>(
    context: context,
    builder: (ctx) => AlertDialog(
      icon: Icon(icon, size: 48, color: color),
      title: Text(title, textAlign: TextAlign.center),
      content: SingleChildScrollView(child: Text(message)),
      actions: [
        FilledButton(onPressed: () => Navigator.pop(ctx), child: Text(okLabel)),
      ],
    ),
  );
}

Future<bool?> showAppConfirm(
  BuildContext context, {
  required String title,
  required String message,
  String cancelLabel = 'Annuler',
  String confirmLabel = 'Continuer',
}) {
  return showDialog<bool>(
    context: context,
    builder: (ctx) => AlertDialog(
      title: Text(title),
      content: Text(message),
      actions: [
        TextButton(onPressed: () => Navigator.pop(ctx, false), child: Text(cancelLabel)),
        FilledButton(onPressed: () => Navigator.pop(ctx, true), child: Text(confirmLabel)),
      ],
    ),
  );
}

/// Inline banner for lists of active alerts.
class AppAlertBanner extends StatelessWidget {
  final String title;
  final String? subtitle;
  final AppAlertType type;

  const AppAlertBanner({
    super.key,
    required this.title,
    this.subtitle,
    this.type = AppAlertType.warning,
  });

  @override
  Widget build(BuildContext context) {
    final (bg, fg, icon) = switch (type) {
      AppAlertType.success => (
          const Color(0xFFE8F8EF),
          const Color(0xFF1E7A3C),
          Icons.check_circle,
        ),
      AppAlertType.warning => (
          const Color(0xFFFFF4E5),
          const Color(0xFFB86E00),
          Icons.warning_amber_rounded,
        ),
      AppAlertType.error => (
          const Color(0xFFFFEBEE),
          const Color(0xFFC62828),
          Icons.error_outline,
        ),
      AppAlertType.info => (
          const Color(0xFFE8F0FF),
          const Color(0xFF2F5BFF),
          Icons.info_outline,
        ),
    };

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: fg.withValues(alpha: 0.35)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: fg, size: 22),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: TextStyle(fontWeight: FontWeight.bold, color: fg)),
                if (subtitle != null) ...[
                  const SizedBox(height: 4),
                  Text(subtitle!, style: TextStyle(fontSize: 13, color: fg.withValues(alpha: 0.9))),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
