import 'package:flutter/material.dart';

class VerificationItem {
  final String label;
  final String? detail;
  final bool passed;
  final bool required;

  const VerificationItem({
    required this.label,
    this.detail,
    required this.passed,
    this.required = true,
  });
}

/// Checklist card used in Vérification tabs.
class VerificationChecklist extends StatelessWidget {
  final List<VerificationItem> items;
  final String? header;
  final String? footer;

  const VerificationChecklist({
    super.key,
    required this.items,
    this.header,
    this.footer,
  });

  bool get allRequiredPassed =>
      items.where((i) => i.required).every((i) => i.passed);

  @override
  Widget build(BuildContext context) {
    final passed = items.where((i) => i.passed).length;
    final total = items.length;

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: BorderSide(color: Colors.grey.shade300),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (header != null) ...[
              Text(header!, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
              const SizedBox(height: 12),
            ],
            Row(
              children: [
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: LinearProgressIndicator(
                      value: total == 0 ? 0 : passed / total,
                      minHeight: 8,
                      backgroundColor: Colors.grey.shade200,
                      color: allRequiredPassed
                          ? const Color(0xFF27AE60)
                          : const Color(0xFFF2994A),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Text('$passed/$total', style: const TextStyle(fontWeight: FontWeight.w600)),
              ],
            ),
            const SizedBox(height: 14),
            ...items.map((item) => _CheckRow(item: item)),
            if (footer != null) ...[
              const SizedBox(height: 12),
              Text(
                footer!,
                style: TextStyle(fontSize: 12, color: Colors.grey.shade700, height: 1.4),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _CheckRow extends StatelessWidget {
  final VerificationItem item;

  const _CheckRow({required this.item});

  @override
  Widget build(BuildContext context) {
    final color = item.passed ? const Color(0xFF27AE60) : const Color(0xFFEB5757);
    final icon = item.passed ? Icons.check_circle : Icons.cancel;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 22),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item.label,
                  style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
                ),
                if (item.detail != null)
                  Text(
                    item.detail!,
                    style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
                  ),
              ],
            ),
          ),
          if (!item.required)
            Text('opt.', style: TextStyle(fontSize: 10, color: Colors.grey.shade500)),
        ],
      ),
    );
  }
}
