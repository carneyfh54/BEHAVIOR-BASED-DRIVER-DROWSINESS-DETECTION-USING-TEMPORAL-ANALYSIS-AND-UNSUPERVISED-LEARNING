import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.first;
  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'John Logistics',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: Brightness.light,
        ),
        primaryColor: Colors.blue,
        secondaryHeaderColor: Colors.green,
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.blue,
          foregroundColor: Colors.white,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blue,
            foregroundColor: Colors.white,
          ),
        ),
      ),
      home: CameraScreen(camera: camera),
    );
  }
}

class CameraScreen extends StatefulWidget {
  final CameraDescription camera;

  const CameraScreen({super.key, required this.camera});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  XFile? _videoFile;
  bool _isRecording = false;
  bool _isInitialized = false;
  String _statusText = "Initializing camera...";
  int _recordingDuration = 0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    setState(() {
      _statusText = "Requesting camera permission...";
    });

    // Request permissions
    final cameraPermission = await Permission.camera.request();
    final micPermission = await Permission.microphone.request();

    if (cameraPermission.isGranted && micPermission.isGranted) {
      setState(() {
        _statusText = "Setting up camera...";
      });

      _controller = CameraController(
        widget.camera,
        ResolutionPreset.high,
        enableAudio: true,
      );

      try {
        await _controller!.initialize();
        
        if (mounted) {
          setState(() {
            _isInitialized = true;
            _statusText = "Ready to record";
          });
        }
      } catch (e) {
        // Handle emulator/模拟器 case where camera might not be available
        if (mounted) {
          setState(() {
            _statusText = "Camera not available on this device/emulator.\nPlease use a real device for camera features.";
          });
        }
      }
    } else {
      setState(() {
        _statusText = "Camera/microphone permission denied";
      });
    }
  }

  Future<void> _startRecording() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      return;
    }

    try {
      await _controller!.startVideoRecording();
      setState(() {
        _isRecording = true;
        _statusText = "🔴 Recording - Detecting driver drowsiness...";
        _recordingDuration = 0;
      });

      // Simulate recording duration counter
      while (_isRecording) {
        await Future.delayed(const Duration(seconds: 1));
        if (mounted && _isRecording) {
          setState(() {
            _recordingDuration++;
          });
        }
      }
    } catch (e) {
      setState(() {
        _statusText = "Error starting recording: $e";
      });
    }
  }

  Future<void> _stopRecording() async {
    if (_controller == null || !_controller!.value.isRecordingVideo) {
      return;
    }

    try {
      final XFile? videoFile = await _controller!.stopVideoRecording();
      setState(() {
        _videoFile = videoFile;
        _isRecording = false;
        _statusText = "Recording saved: ${videoFile?.path ?? 'Unknown'}";
      });
    } catch (e) {
      setState(() {
        _statusText = "Error stopping recording: $e";
        _isRecording = false;
      });
    }
  }

  String _formatDuration(int seconds) {
    final minutes = seconds ~/ 60;
    final secs = seconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${secs.toString().padLeft(2, '0')}';
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return Scaffold(
        body: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [Colors.blue, Colors.lightBlueAccent],
            ),
          ),
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const CircularProgressIndicator(color: Colors.white),
                const SizedBox(height: 20),
                Text(
                  _statusText,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return Scaffold(
      body: Stack(
        children: [
          // Camera Preview
          SizedBox.expand(
            child: CameraPreview(_controller!),
          ),

          // Text Overlay - Top Section
          Positioned(
            top: 50,
            left: 0,
            right: 0,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.black.withOpacity(0.7),
                    Colors.transparent,
                  ],
                ),
              ),
              child: Column(
                children: [
                  Text(
                    _statusText,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      shadows: [
                        Shadow(
                          color: Colors.black,
                          offset: Offset(1, 1),
                          blurRadius: 3,
                        ),
                      ],
                    ),
                  ),
                  if (_isRecording) ...[
                    const SizedBox(height: 8),
                    Text(
                      "Duration: ${_formatDuration(_recordingDuration)}",
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontFamily: 'monospace',
                        shadows: [
                          Shadow(
                            color: Colors.black,
                            offset: Offset(1, 1),
                            blurRadius: 3,
                          ),
                        ],
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),

          // Additional Info Overlay - Middle Section
          Positioned(
            top: 150,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.5),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Column(
                  children: [
                    Text(
                      "DRIVER DROWSINESS DETECTION",
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 1.2,
                      ),
                    ),
                    SizedBox(height: 5),
                    Text(
                      "Video analysis in progress",
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Recording Indicator
          if (_isRecording)
            Positioned(
              top: 50,
              right: 20,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.red,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: const BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 6),
                    const Text(
                      "REC",
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),

          // Control Buttons - Bottom Section
          Positioned(
            bottom: 30,
            left: 0,
            right: 0,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                // Recording Info Button
                FloatingActionButton.small(
                  onPressed: () {
                    showDialog(
                      context: context,
                      builder: (context) => AlertDialog(
                        title: const Text("Recording Info"),
                        content: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text("Status: ${_isRecording ? 'Recording' : 'Idle'}"),
                            if (_videoFile != null)
                              Text("Last recording: ${_videoFile!.path.split('/').last}"),
                          ],
                        ),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.pop(context),
                            child: const Text("Close"),
                          ),
                        ],
                      ),
                    );
                  },
                  backgroundColor: Colors.white.withOpacity(0.3),
                  child: const Icon(Icons.info, color: Colors.white),
                ),

                // Record/Stop Button
                FloatingActionButton(
                  onPressed: _isRecording ? _stopRecording : _startRecording,
                  backgroundColor: _isRecording ? Colors.red : Colors.white,
                  foregroundColor: _isRecording ? Colors.white : Colors.red,
                  child: Icon(
                    _isRecording ? Icons.stop : Icons.fiber_manual_record,
                    size: 32,
                  ),
                ),

                // Switch Camera Button
                FloatingActionButton.small(
                  onPressed: () async {
                    if (_controller != null && _isInitialized) {
                      final cameras = await availableCameras();
                      final currentDescription = _controller!.description;
                      final newCamera = cameras.firstWhere(
                        (camera) => camera.lensDirection != currentDescription.lensDirection,
                      );
                      
                      await _controller!.dispose();
                      _controller = CameraController(
                        newCamera,
                        ResolutionPreset.high,
                        enableAudio: true,
                      );
                      await _controller!.initialize();
                      
                      if (mounted) {
                        setState(() {});
                      }
                    }
                  },
                  backgroundColor: Colors.white.withOpacity(0.3),
                  child: const Icon(Icons.cameraswitch, color: Colors.white),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
