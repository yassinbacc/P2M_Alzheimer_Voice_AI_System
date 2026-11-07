# Alzheimer Detection App

An AI-powered mobile application for early detection of Alzheimer's disease predisposition using voice biomarkers and natural language processing.

## Project Overview

This Flutter application implements a complete solution for:
- **Voice Biomarker Analysis**: Records and analyzes voice patterns using NLP
- **Patient Profile Management**: Securely stores patient information
- **Assessment History**: Tracks risk scores over time
- **Hybrid Deployment**: Supports both on-device and cloud-based processing
- **Cognitive Assessment**: AI-based prediction using voice patterns

## Features

- 🎤 **Audio Recording**: High-quality voice recording (16-bit, 16kHz)
- 📊 **Risk Assessment**: AI-powered Alzheimer's risk scoring (0-100%)
- 👤 **Patient Profiles**: Complete patient information management
- 📈 **Assessment Trends**: Visual trends and historical analysis
- 🔒 **Data Security**: Local storage with encryption support
- 📱 **Mobile First**: iOS and Android support
- 🌐 **Hybrid Processing**: On-device and cloud analysis options

## Architecture

```
lib/
├── main.dart                 # App entry point
├── screens/                  # UI Screens
│   ├── home_screen.dart
│   ├── patient_profile_screen.dart
│   ├── audio_recording_screen.dart
│   ├── assessment_results_screen.dart
│   ├── patient_history_screen.dart
│   └── settings_screen.dart
├── models/                   # Data models
│   ├── patient.dart
│   ├── assessment.dart
│   └── app_state.dart
├── services/                 # Business logic
│   ├── api_service.dart      # Backend API calls
│   ├── audio_service.dart    # Audio recording
│   └── database_service.dart # Local database
├── widgets/                  # Reusable components
├── utils/                    # Utilities
└── assets/                   # Images and resources
```

## Tech Stack

- **Flutter 3.0+**: Cross-platform mobile framework
- **Dart**: Programming language
- **Dio**: HTTP client for API calls
- **SQLite**: Local database
- **Provider/Riverpod**: State management
- **record**: Audio recording library
- **intl**: Internationalization

## Project Structure

### Models
- **Patient**: User profile information (name, DOB, gender, medical history)
- **Assessment**: Audio assessment results with risk scores
- **AlzheimersRiskScore**: Risk assessment with biomarkers

### Services
- **ApiService**: REST API communication for:
  - Patient CRUD operations
  - Assessment submission and analysis
  - Health checks
  
- **AudioService**: Recording management:
  - Start/stop recording
  - Duration tracking
  - File management
  
- **DatabaseService**: SQLite integration:
  - Local patient data
  - Assessment history
  - Offline capability

### Screens
1. **HomeScreen**: Dashboard with quick actions
2. **PatientProfileScreen**: Edit patient information
3. **AudioRecordingScreen**: Record voice samples
4. **AssessmentResultsScreen**: Display analysis results
5. **PatientHistoryScreen**: View assessment history and trends
6. **SettingsScreen**: App configuration

## Getting Started

### Prerequisites
- Flutter SDK (v3.0.0 or higher)
- Dart SDK (bundled with Flutter)
- Android Studio or Xcode for emulators
- Visual Studio Code with Flutter extension

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd alzheimer_detection
```

2. **Install dependencies**
```bash
flutter pub get
```

3. **Run the app**
```bash
flutter run
```

### Development Commands

```bash
# Analyze code
flutter analyze

# Format code
dart format lib/

# Run tests
flutter test

# Build APK (Android)
flutter build apk

# Build IPA (iOS)
flutter build ios

# Clean build
flutter clean
```

## API Integration

The app communicates with a backend API for:
- Patient data persistence
- Audio analysis and processing
- Risk score calculation

### API Endpoints (Expected)
```
GET    /api/health         - Health check
GET    /api/patients       - List patients
POST   /api/patients       - Create patient
PUT    /api/patients/{id}  - Update patient
DELETE /api/patients/{id}  - Delete patient

GET    /api/assessments    - List assessments
POST   /api/assessments    - Save assessment
POST   /api/assessments/analyze - Analyze audio
PUT    /api/assessments/{id}    - Update assessment
DELETE /api/assessments/{id}    - Delete assessment
```

## Database Schema

### Patients Table
```sql
- id (TEXT, PRIMARY KEY)
- firstName (TEXT)
- lastName (TEXT)
- dateOfBirth (TEXT)
- gender (TEXT)
- email (TEXT)
- phoneNumber (TEXT)
- medicalHistory (TEXT)
- createdAt (TEXT)
- updatedAt (TEXT)
```

### Assessments Table
```sql
- id (TEXT, PRIMARY KEY)
- patientId (TEXT, FOREIGN KEY)
- audioFilePath (TEXT)
- recordedAt (TEXT)
- duration (INTEGER)
- transcription (TEXT)
- riskScore (REAL)
- riskLevel (TEXT)
- confidence (REAL)
- biomarkers (TEXT)
- analysisDetails (TEXT)
- notes (TEXT)
- createdAt (TEXT)
```

## Risk Assessment Levels

- **Low Risk** (0-30%): Normal cognitive patterns
- **Moderate Risk** (30-70%): Some cognitive decline indicators
- **High Risk** (70-100%): Significant cognitive decline indicators

## Audio Specifications

- **Sample Rate**: 16,000 Hz
- **Bit Depth**: 16-bit
- **Format**: WAV
- **Duration**: 2-3 minutes recommended
- **Channels**: Mono

## Security Considerations

1. **Data Encryption**: All sensitive data encrypted locally
2. **API Security**: HTTPS for all network requests
3. **User Authentication**: Optional authentication layer
4. **Privacy**: GDPR-compliant data handling
5. **Permissions**: Audio recording requires user consent

## State Management

Currently uses **Provider** pattern with **Riverpod** for future migration. Consider:
- BLoC pattern for complex features
- Riverpod for reactive programming
- GetX for simpler state management

## Future Enhancements

- [ ] Multi-language support (French, Arabic, Spanish)
- [ ] Biometric authentication
- [ ] Cloud synchronization
- [ ] Offline mode improvements
- [ ] Advanced ML models on-device
- [ ] Real-time audio waveform visualization
- [ ] Doctor consultation integration
- [ ] Family member sharing
- [ ] Wearable device integration

## Testing

```bash
# Run unit tests
flutter test test/

# Generate coverage
flutter test --coverage
```

## Deployment

### Android
```bash
flutter build apk --release
flutter build appbundle --release
```

### iOS
```bash
flutter build ios --release
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Format code: `dart format lib/`
4. Run tests
5. Submit pull request

## License

Proprietary - Ouest Technologies & Supcom 3IT

## Contact & Support

For support or questions, contact:
- Email: Sofia.benjebara@supcom.tn
- Department: MASC (Supcom)

## Disclaimer

⚠️ **Medical Disclaimer**: This application is for research and educational purposes only. It is not a medical device and should not be used as a substitute for professional medical diagnosis. Always consult with a qualified healthcare professional for evaluation and diagnosis.

## Changelog

### Version 1.0.0 (Initial Release)
- Initial project setup
- Home screen with navigation
- Patient profile management
- Audio recording functionality
- Assessment results display
- Assessment history and trends
- Settings and configuration
- Local database integration
- API service integration
