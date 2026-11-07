# Alzheimer Detection App - Development Instructions

This is a Flutter mobile application for early detection of Alzheimer's disease using voice biomarkers and AI analysis.

## Project Information

- **Project Type**: Flutter Mobile App (Cross-platform - iOS & Android)
- **Language**: Dart
- **Framework**: Flutter 3.0+
- **Purpose**: AI-powered Alzheimer's disease detection using voice analysis

## Architecture Overview

```
Presentation Layer (Screens)
    ↓
Business Logic Layer (Services)
    ↓
Data Layer (Models, Database, API)
```

## Key Directories

- `lib/` - Source code
  - `screens/` - UI screens and pages
  - `models/` - Data models
  - `services/` - Business logic (API, Audio, Database)
  - `widgets/` - Reusable UI components
  - `utils/` - Utility functions
- `assets/` - Images, fonts, and resources
- `test/` - Unit and widget tests

## Dependencies

Core dependencies in `pubspec.yaml`:
- **record**: Audio recording functionality
- **dio**: HTTP client for API calls
- **sqflite**: Local SQLite database
- **provider/riverpod**: State management
- **audio_waveforms**: Audio visualization
- **path_provider**: File system access

## Development Workflow

1. **Running the App**
   ```
   flutter run
   ```

2. **Hot Reload** (during development)
   ```
   Press 'r' in terminal after making code changes
   ```

3. **Code Formatting**
   ```
   dart format lib/
   ```

4. **Analyze Code**
   ```
   flutter analyze
   ```

## Key Features to Implement

### Current Phase - Frontend UI Complete
- [x] Home screen with navigation
- [x] Patient profile management UI
- [x] Audio recording UI
- [x] Assessment results display UI
- [x] Assessment history UI
- [x] Settings UI

### Next Phase - Backend Integration
- [ ] Connect to backend API
- [ ] Implement actual audio recording
- [ ] Database persistence
- [ ] State management refactoring

### Future Phases
- [ ] ML model integration
- [ ] Advanced audio processing
- [ ] Real-time waveform visualization
- [ ] Cloud synchronization

## Code Standards

- Follow Dart style guide
- Use PascalCase for classes
- Use camelCase for variables/methods
- Use descriptive names
- Add comments for complex logic
- Keep widgets focused and reusable

## Screens Architecture

Each screen implements a StatefulWidget with:
1. UI layout with proper spacing
2. Form validation where needed
3. Navigation using Navigator
4. Mock data for demonstration

### Screen Details

- **HomeScreen**: Dashboard with quick actions and feature info
- **PatientProfileScreen**: Form-based patient data entry
- **AudioRecordingScreen**: Audio capture with timer
- **AssessmentResultsScreen**: Results display with biomarkers
- **PatientHistoryScreen**: Assessment history with trends
- **SettingsScreen**: App configuration and account settings

## Services Architecture

### ApiService
- REST API communication
- Patient CRUD operations
- Assessment submission
- Audio file upload

### AudioService
- Recording start/stop
- Duration tracking
- File management
- Permission handling

### DatabaseService
- SQLite integration
- Local data persistence
- Query operations

## Data Models

### Patient
- Personal information
- Medical history
- Profile management

### Assessment
- Audio file reference
- Risk scores
- Voice biomarkers
- Analysis results
- Timestamps

### AlzheimersRiskScore
- Risk percentage (0-1.0)
- Risk level (Low/Moderate/High)
- Confidence score
- Detected biomarkers

## Next Developer Steps

1. Set up backend API (Python/Node.js)
2. Implement actual audio recording with `record` package
3. Connect ApiService to real endpoints
4. Implement state management (Provider/Riverpod)
5. Add unit and widget tests
6. Integrate ML model for risk assessment
7. Add authentication
8. Implement cloud synchronization

## Common Tasks

### Adding a New Screen
1. Create file in `lib/screens/`
2. Create StatefulWidget/StatelessWidget
3. Add route in `main.dart`
4. Add navigation from existing screens

### Adding a New Service
1. Create file in `lib/services/`
2. Implement service logic
3. Create singleton instance or use DI
4. Use from screens via Provider/Riverpod

### Adding a New Model
1. Create file in `lib/models/`
2. Include JSON serialization (toJson/fromJson)
3. Implement copyWith for immutability
4. Use in services and screens

## Testing

- Run widget tests: `flutter test`
- Check code coverage: `flutter test --coverage`

## Debugging

- Use Flutter DevTools: `flutter pub global run devtools`
- Check logs: `flutter logs`
- Use breakpoints in IDE

## Deployment

Android:
```
flutter build apk --release
flutter build appbundle --release
```

iOS:
```
flutter build ios --release
```

## Resources

- Flutter Docs: https://flutter.dev/docs
- Dart Docs: https://dart.dev/guides
- Package Documentation: Check pubspec.yaml for links

## Contact

- Project Lead: BEN JEBARA Sofia
- Department: MASC (Supcom)
- Email: Sofia.benjebara@supcom.tn
