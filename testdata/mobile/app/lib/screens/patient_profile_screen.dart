import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/patient.dart';
import '../services/local_db.dart';
import '../services/session_service.dart';

class PatientProfileScreen extends StatefulWidget {
  final Patient patient;

  const PatientProfileScreen({Key? key, required this.patient}) : super(key: key);

  @override
  State<PatientProfileScreen> createState() => _PatientProfileScreenState();
}

class _PatientProfileScreenState extends State<PatientProfileScreen> {
  final _formKey = GlobalKey<FormState>();
  final SessionService _session = SessionService();
  final LocalDb _db = LocalDb();

  late final TextEditingController _firstNameController;
  late final TextEditingController _lastNameController;
  late final TextEditingController _emailController;
  late final TextEditingController _phoneController;
  late final TextEditingController _medicalHistoryController;

  late DateTime _selectedDate;
  late String _selectedGender;

  @override
  void initState() {
    super.initState();
    final p = widget.patient;
    _firstNameController = TextEditingController(text: p.firstName);
    _lastNameController = TextEditingController(text: p.lastName);
    _emailController = TextEditingController(text: p.email);
    _phoneController = TextEditingController(text: p.phoneNumber);
    _medicalHistoryController = TextEditingController(text: p.medicalHistory);
    _selectedDate = p.dateOfBirth;
    _selectedGender = p.gender.isNotEmpty ? p.gender : 'Non renseigné';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profil patient', style: TextStyle(fontWeight: FontWeight.bold)),
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              CircleAvatar(
                radius: 48,
                backgroundColor: const Color(0xFF2F5BFF).withOpacity(0.15),
                child: Text(
                  widget.patient.firstName.isNotEmpty
                      ? widget.patient.firstName[0].toUpperCase()
                      : '?',
                  style: const TextStyle(
                      fontSize: 36, fontWeight: FontWeight.bold, color: Color(0xFF2F5BFF)),
                ),
              ),
              const SizedBox(height: 8),
              Text('ID: ${widget.patient.id}', style: const TextStyle(color: Colors.black54)),
              const SizedBox(height: 24),
              TextFormField(
                controller: _firstNameController,
                decoration: const InputDecoration(labelText: 'Prénom'),
                validator: (v) => v == null || v.trim().isEmpty ? 'Requis' : null,
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _lastNameController,
                decoration: const InputDecoration(labelText: 'Nom'),
              ),
              const SizedBox(height: 12),
              ListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Date de naissance'),
                subtitle: Text(
                    '${_selectedDate.day}/${_selectedDate.month}/${_selectedDate.year}'),
                trailing: const Icon(Icons.calendar_today),
                onTap: () async {
                  final picked = await showDatePicker(
                    context: context,
                    initialDate: _selectedDate,
                    firstDate: DateTime(1900),
                    lastDate: DateTime.now(),
                  );
                  if (picked != null) setState(() => _selectedDate = picked);
                },
              ),
              const SizedBox(height: 12),
              DropdownButtonFormField<String>(
                value: _selectedGender,
                decoration: const InputDecoration(labelText: 'Genre'),
                items: ['Homme', 'Femme', 'Autre', 'Non renseigné']
                    .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                    .toList(),
                onChanged: (v) => setState(() => _selectedGender = v ?? 'Non renseigné'),
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _emailController,
                decoration: const InputDecoration(labelText: 'Email'),
                keyboardType: TextInputType.emailAddress,
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _phoneController,
                decoration: const InputDecoration(labelText: 'Téléphone'),
                keyboardType: TextInputType.phone,
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _medicalHistoryController,
                decoration: const InputDecoration(labelText: 'Antécédents médicaux'),
                maxLines: 4,
              ),
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _save,
                  child: const Text('Enregistrer'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    final updated = Patient(
      id: widget.patient.id,
      firstName: _firstNameController.text.trim(),
      lastName: _lastNameController.text.trim(),
      dateOfBirth: _selectedDate,
      gender: _selectedGender,
      email: _emailController.text.trim(),
      phoneNumber: _phoneController.text.trim(),
      medicalHistory: _medicalHistoryController.text.trim(),
      createdAt: widget.patient.createdAt,
      updatedAt: DateTime.now(),
    );
    await _session.savePatient(updated);
    if (!kIsWeb) {
      try {
        await _db.updatePatient(updated);
      } catch (e) {
        debugPrint('Profile DB update: $e');
      }
    }
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Profil mis à jour')),
      );
      Navigator.pop(context);
    }
  }

  @override
  void dispose() {
    _firstNameController.dispose();
    _lastNameController.dispose();
    _emailController.dispose();
    _phoneController.dispose();
    _medicalHistoryController.dispose();
    super.dispose();
  }
}
