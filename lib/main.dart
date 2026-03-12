import 'dart:async';
import 'dart:typed_data';
import 'face_mesh_painter.dart';
import 'web_camera_service.dart';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;
import 'video_analysis_service.dart';

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
      title: 'Driver Drowsiness Detection',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue, brightness: Brightness.light),
        primaryColor: Colors.blue,
        appBarTheme: const AppBarTheme(backgroundColor: Colors.blue, foregroundColor: Colors.white),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(backgroundColor: Colors.blue, foregroundColor: Colors.white),
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
  bool _isRecording = false;
  bool _isInitialized = false;
  bool _isConnected = false;
  String _statusText = "Initializing camera...";
  String _connectionStatus = "Disconnected";
  String? _connectionError;
  int _recordingDuration = 0;
  int _framesSent = 0;
  Timer? _frameCaptureTimer;

  AnalysisResult? _currentAnalysis;
  List<FaceLandmark> _currentLandmarks = [];

  String _serverUrl = 'localhost';
  final TextEditingController _serverUrlController = TextEditingController();

  final VideoAnalysisService _videoService = VideoAnalysisService();
  final WebCameraService _webCameraService = WebCameraService();

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _setupVideoServiceCallbacks();
  }

  void _setupVideoServiceCallbacks() {
    _videoService.setCallbacks(
      onMessage: (type, data) {
        if (mounted) {
          setState(() {
            if (type == 'connection_established') {
              _isConnected = true;
              _connectionStatus = "Connected";
              _connectionError = null;
              _statusText = "Connected - Ready to analyze";
            } else if (type == 'analysis_result') {
              try {
                final result = VideoAnalysisService.parseAnalysisResult(data);
                _currentAnalysis = result;
                _currentLandmarks = result.landmarks;
              } catch (e) {
                if (kDebugMode) print('Error parsing result: $e');
              }
            }
          });
        }
      },
      onConnected: () {
        if (mounted) setState(() {
          _isConnected = true;
          _connectionStatus = "Connected";
          _connectionError = null;
        });
      },
      onError: (error) {
        if (mounted) setState(() {
          _isConnected = false;
          _connectionStatus = "Error";
          _connectionError = error;
          _statusText = "Connection error: $error";
        });
      },
      onDisconnected: () {
        if (mounted) setState(() {
          _isConnected = false;
          _connectionStatus = "Disconnected";
        });
      },
    );
  }

  Future<void> _initializeCamera() async {
    setState(() => _statusText = "Requesting camera permission...");

    if (!kIsWeb) {
      final cameraPermission = await Permission.camera.request();
      final micPermission = await Permission.microphone.request();
      if (!cameraPermission.isGranted || !micPermission.isGranted) {
        setState(() => _statusText = "Camera permission denied");
        return;
      }
    }

    setState(() => _statusText = "Setting up camera...");
    _controller = CameraController(widget.camera, ResolutionPreset.medium, enableAudio: false);

    try {
      await _controller!.initialize();
      if (mounted) setState(() {
        _isInitialized = true;
        _statusText = "Ready to start real-time analysis";
      });
    } catch (e) {
      if (mounted) setState(() => _statusText = "Camera error: $e");
    }
  }

  Future<void> _startAnalysis() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    try {
      await _videoService.connect(_serverUrl, port: 8000);

      setState(() {
        _isRecording = true;
        _statusText = "🔴 Analyzing drowsiness in real-time...";
        _recordingDuration = 0;
        _framesSent = 0;
      });

      _startDurationCounter();

      if (kIsWeb) {
        await Future.delayed(const Duration(milliseconds: 600));
        await _webCameraService.initialize();
        _frameCaptureTimer = Timer.periodic(const Duration(milliseconds: 300), (_) async {
          if (!_isRecording || !_videoService.isConnected) return;
          try {
            final bytes = await _webCameraService.captureFrame();
            if (bytes != null && bytes.isNotEmpty) {
              await _videoService.sendFrame(bytes);
              setState(() => _framesSent++);
            }
          } catch (e) {
            if (kDebugMode) print('Web frame error: $e');
          }
        });
      } else {
        await _controller!.startImageStream(_captureFrame);
      }

    } catch (e) {
      setState(() {
        _statusText = "Failed to start: $e";
        _connectionError = e.toString();
      });
    }
  }

  void _startDurationCounter() {
    Timer.periodic(const Duration(seconds: 1), (timer) {
      if (mounted && _isRecording) {
        setState(() => _recordingDuration++);
      } else {
        timer.cancel();
      }
    });
  }

  Future<void> _captureFrame(CameraImage image) async {
    if (!_isRecording || !_videoService.isConnected) return;
    try {
      final bytes = await _convertToJpeg(image);
      if (bytes.isNotEmpty) {
        await _videoService.sendFrame(bytes);
        setState(() => _framesSent++);
      }
    } catch (e) {
      if (kDebugMode) print('Frame error: $e');
    }
  }

  Future<Uint8List> _convertToJpeg(CameraImage image) async {
    try {
      img.Image? converted;
      if (image.format.group == ImageFormatGroup.yuv420) {
        converted = img.Image(width: image.width, height: image.height);
        final yPlane = image.planes[0];
        final uPlane = image.planes[1];
        final vPlane = image.planes[2];
        final yBuf = yPlane.bytes;
        final uBuf = uPlane.bytes;
        final vBuf = vPlane.bytes;
        final yStride = yPlane.bytesPerRow;
        final uvStride = uPlane.bytesPerRow;
        final uvPixel = uPlane.bytesPerPixel ?? 1;
        for (int y = 0; y < image.height; y++) {
          for (int x = 0; x < image.width; x++) {
            final yVal = yBuf[y * yStride + x];
            final uvIdx = (y ~/ 2) * uvStride + (x ~/ 2) * uvPixel;
            final uVal = uBuf[uvIdx] - 128;
            final vVal = vBuf[uvIdx] - 128;
            final r = (yVal + 1.402 * vVal).round().clamp(0, 255);
            final g = (yVal - 0.344136 * uVal - 0.714136 * vVal).round().clamp(0, 255);
            final b = (yVal + 1.772 * uVal).round().clamp(0, 255);
            converted.setPixelRgba(x, y, r, g, b, 255);
          }
        }
      } else if (image.format.group == ImageFormatGroup.bgra8888) {
        converted = img.Image.fromBytes(
          width: image.width, height: image.height,
          bytes: image.planes[0].bytes.buffer, order: img.ChannelOrder.bgra,
        );
      } else if (image.format.group == ImageFormatGroup.jpeg) {
        return image.planes[0].bytes;
      }
      if (converted == null) return Uint8List(0);
      return Uint8List.fromList(img.encodeJpg(converted, quality: 70));
    } catch (e) {
      return Uint8List(0);
    }
  }

  Future<void> _stopAnalysis() async {
    _frameCaptureTimer?.cancel();
    _frameCaptureTimer = null;
    if (!kIsWeb && _controller != null && _controller!.value.isStreamingImages) {
      await _controller!.stopImageStream();
    }
    _videoService.disconnect();
    setState(() {
      _isRecording = false;
      _isConnected = false;
      _connectionStatus = "Disconnected";
      _statusText = "Analysis stopped";
      _currentAnalysis = null;
      _currentLandmarks = [];
      _currentAnalysis = null;
      _currentLandmarks = [];
    });
  }

  Future<void> _connectToServer() async {
    setState(() { _connectionStatus = "Connecting..."; _connectionError = null; });
    try {
      await _videoService.connect(_serverUrl, port: 8000);
      if (_videoService.isConnected) setState(() => _statusText = "Connected! Press Start");
    } catch (e) {
      setState(() { _connectionError = "Failed: $e"; _connectionStatus = "Failed"; });
    }
  }

  String _formatDuration(int s) =>
      '${(s ~/ 60).toString().padLeft(2, '0')}:${(s % 60).toString().padLeft(2, '0')}';

  @override
  void dispose() {
    _frameCaptureTimer?.cancel();
    _controller?.dispose();
    _videoService.disconnect();
    _serverUrlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return Scaffold(
        body: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter, end: Alignment.bottomCenter,
              colors: [Colors.blue, Colors.lightBlueAccent],
            ),
          ),
          child: Center(child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(color: Colors.white),
              const SizedBox(height: 20),
              Text(_statusText, style: const TextStyle(color: Colors.white, fontSize: 16)),
            ],
          )),
        ),
      );
    }

    return Scaffold(
      body: Stack(
        children: [
          // Camera preview
          SizedBox.expand(child: CameraPreview(_controller!)),

          // Face mesh overlay
          if (_currentLandmarks.isNotEmpty)
            Positioned.fill(
              child: CustomPaint(
                painter: FaceMeshPainter(
                  landmarks: _currentLandmarks,
                  drowsinessLevel: _currentAnalysis?.drowsinessLevel ?? 'awake',
                ),
              ),
            ),

          // Connection badge
          Positioned(
            top: 48, left: 10, right: 10,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: _isConnected ? Colors.green.withValues(alpha: 0.85) : Colors.orange.withValues(alpha: 0.85),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Row(mainAxisSize: MainAxisSize.min, children: [
                    Icon(_isConnected ? Icons.cloud_done : Icons.cloud_off, color: Colors.white, size: 14),
                    const SizedBox(width: 5),
                    Text(_connectionStatus, style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold)),
                  ]),
                ),
              ],
            ),
          ),

          // Analysis result overlay
          if (_currentAnalysis != null)
            Positioned(
              top: 90, left: 10, right: 10,
              child: _buildResultCard(),
            ),

          // Status text
          if (_isRecording)
            Positioned(
              bottom: 110, left: 0, right: 0,
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.5),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    "${_formatDuration(_recordingDuration)} · $_framesSent frames sent",
                    style: const TextStyle(color: Colors.white70, fontSize: 11, fontFamily: 'monospace'),
                  ),
                ),
              ),
            ),

          // Controls
          Positioned(
            bottom: 36, left: 0, right: 0,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                FloatingActionButton.small(
                  heroTag: 'connect',
                  onPressed: _showServerDialog,
                  backgroundColor: _isConnected ? Colors.green : Colors.white.withValues(alpha: 0.3),
                  child: Icon(_isConnected ? Icons.link : Icons.link_off, color: Colors.white),
                ),
                FloatingActionButton(
                  heroTag: 'start',
                  onPressed: _isRecording ? _stopAnalysis : _startAnalysis,
                  backgroundColor: _isRecording ? Colors.red : Colors.white,
                  foregroundColor: _isRecording ? Colors.white : Colors.red,
                  child: Icon(_isRecording ? Icons.stop : Icons.play_arrow, size: 32),
                ),
                FloatingActionButton.small(
                  heroTag: 'settings',
                  onPressed: _showServerDialog,
                  backgroundColor: Colors.white.withValues(alpha: 0.3),
                  child: const Icon(Icons.settings, color: Colors.white),
                ),
              ],
            ),
          ),

          if (_connectionError != null) _buildErrorDialog(),
        ],
      ),
    );
  }

  Widget _buildResultCard() {
    final analysis = _currentAnalysis!;
    final colors = {
      'awake': Colors.green,
      'mildly drowsy': Colors.yellow,
      'moderately drowsy': Colors.orange,
      'highly drowsy': Colors.red,
    };
    final color = colors[analysis.drowsinessLevel] ?? Colors.grey;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.7),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color, width: 2),
      ),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(analysis.isDrowsy ? Icons.warning : Icons.check_circle, color: color, size: 22),
          const SizedBox(width: 8),
          Text(analysis.drowsinessStatus, style: TextStyle(color: color, fontSize: 15, fontWeight: FontWeight.bold)),
        ]),
        const SizedBox(height: 6),
        Text("Confidence: ${(analysis.confidence * 100).toStringAsFixed(1)}%",
            style: const TextStyle(color: Colors.white70, fontSize: 12)),
        if (analysis.observations.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              analysis.observations.first,
              style: const TextStyle(color: Colors.white60, fontSize: 10),
              textAlign: TextAlign.center,
            ),
          ),
      ]),
    );
  }

  void _showServerDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Server Configuration"),
        content: Column(mainAxisSize: MainAxisSize.min, children: [
          TextField(
            controller: _serverUrlController,
            decoration: const InputDecoration(
              labelText: "Server IP",
              hintText: "localhost or 172.20.10.5",
              prefixIcon: Icon(Icons.dns),
            ),
            onChanged: (v) => _serverUrl = v,
          ),
          const SizedBox(height: 16),
          SizedBox(width: double.infinity,
            child: ElevatedButton(
              onPressed: () { Navigator.pop(context); _connectToServer(); },
              child: const Text("Connect"),
            ),
          ),
          if (_connectionError != null) ...[
            const SizedBox(height: 10),
            Text(_connectionError!, style: const TextStyle(color: Colors.red, fontSize: 12)),
          ],
        ]),
        actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text("Close"))],
      ),
    );
  }

  Widget _buildErrorDialog() {
    return AlertDialog(
      title: const Text("Error"),
      content: Text(_connectionError ?? "Unknown error"),
      actions: [
        TextButton(onPressed: () => setState(() => _connectionError = null), child: const Text("Dismiss")),
        TextButton(onPressed: () { setState(() => _connectionError = null); _connectToServer(); }, child: const Text("Retry")),
      ],
    );
  }
}
