import 'package:flutter_test/flutter_test.dart';

import 'package:alzheimer_detection/models/patient.dart';
import 'package:alzheimer_detection/screens/home_screen.dart';

void main() {
  testWidgets('Home screen shows patient name', (WidgetTester tester) async {
    final patient = Patient(
      id: 'P001',
      firstName: 'Jean',
      lastName: 'Dupont',
      dateOfBirth: DateTime(1950, 1, 1),
      gender: 'Homme',
      email: '',
      phoneNumber: '',
      medicalHistory: '',
      createdAt: DateTime.now(),
      updatedAt: DateTime.now(),
    );

    await tester.pumpWidget(HomeScreen(patient: patient));
    await tester.pump();

    expect(find.text('Jean'), findsOneWidget);
    expect(find.text('Nouveau Test'), findsOneWidget);
  });
}
