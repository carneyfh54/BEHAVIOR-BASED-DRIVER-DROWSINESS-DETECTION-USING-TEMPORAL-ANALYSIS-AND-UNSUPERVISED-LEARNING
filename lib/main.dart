import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
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
  bool _isRecording = false;
  bool _isInitialized = false;
  bool _isConnected = false;
  String _statusText = "Initializing camera...";
  String _connectionStatus = "Disconnected";
  String? _connectionError;
  int _recordingDuration = 0;
  
  // Video streaming
  Timer? _frameCaptureTimer;
  int _frameCount = 0;
  int _framesSent = 0;
  
  // AI Analysis results
  AnalysisResult? _currentAnalysis;
  List<AnalysisResult> _analysisHistory = [];
  bool _isAnalyzing = false;
  
  // Server configuration
  String _serverUrl = 'localhost';
  final TextEditingController _serverUrlController = TextEditingController();
  
  final VideoAnalysisService _videoService = VideoAnalysisService();

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
              _isAnalyzing = false;
              try {
                final result = VideoAnalysisService.parseAnalysisResult(data);
                _currentAnalysis = result;
                _analysisHistory.insert(0, result);
                if (_analysisHistory.length > 10) {
                  _analysisHistory.removeLast();
                }
              } catch (e) {
                if (kDebugMode) {
                  print('Error parsing analysis result: $e');
                }
              }
            } else if (type == 'processing') {
              _isAnalyzing = true;
            } else if (type == 'frame_received') {
              if (data['analyzed'] == false) {
                _framesSent++;
              }
            }
          });
        }
      },
      onConnected: () {
        if (mounted) {
          setState(() {
            _isConnected = true;
            _connectionStatus = "Connected";
            _connectionError = null;
          });
        }
      },
      onError: (error) {
        if (mounted) {
          setState(() {
            _isConnected = false;
            _connectionStatus = "Error";
            _connectionError = error;
            _statusText = "Connection error: $error";
          });
        }
      },
      onDisconnected: () {
        if (mounted) {
          setState(() {
            _isConnected = false;
            _connectionStatus = "Disconnected";
          });
        }
      },
    );
  }

  Future<void> _initializeCamera() async {
    setState(() {
      _statusText = "Requesting camera permission...";
    });

    final cameraPermission = await Permission.camera.request();
    final micPermission = await Permission.microphone.request();

    if (cameraPermission.isGranted && micPermission.isGranted) {
      setState(() {
        _statusText = "Setting up camera...";
      });

      _controller = CameraController(
        widget.camera,
        ResolutionPreset.high,
        enableAudio: false, // Disable audio for frame analysis
      );

      try {
        await _controller!.initialize();
        
        if (mounted) {
          setState(() {
            _isInitialized = true;
            _statusText = "Ready to start real-time analysis";
          });
        }
      } catch (e) {
        if (mounted) {
          setState(() {
            _statusText = "Camera not available. Please use a real device.";
          });
        }
      }
    } else {
      setState(() {
        _statusText = "Camera/microphone permission denied";
      });
    }
  }

  Future<void> _startAnalysis() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      return;
    }

    try {
      // Connect to WebSocket server
      await _videoService.connect(_serverUrl, port: 8000);
      
      // Start video streaming
      await _controller!.startImageStream(_captureFrame);
      
      setState(() {
        _isRecording = true;
        _statusText = "🔴 Analyzing driver drowsiness in real-time...";
        _recordingDuration = 0;
      });
      
      // Start duration counter
      _startDurationCounter();
      
    } catch (e) {
      setState(() {
        _statusText = "Failed to start analysis: $e";
        _connectionError = e.toString();
      });
    }
  }
  
  void _startDurationCounter() {
    Timer.periodic(const Duration(seconds: 1), (timer) {
      if (mounted && _isRecording) {
        setState(() {
          _recordingDuration++;
        });
      } else {
        timer.cancel();
      }
    });
  }

  Future<void> _captureFrame(CameraImage image) async {
    if (!_isRecording || !_videoService.isConnected) {
      return;
    }
    
    try {
      // Convert CameraImage to JPEG bytes
      final Uint8List jpegBytes = await _convertCameraImageToJpeg(image);
      
      if (jpegBytes.isNotEmpty) {
        _frameCount++;
        await _videoService.sendFrame(jpegBytes);
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error capturing frame: $e');
      }
    }
  }
  
  Future<Uint8List> _convertCameraImageToJpeg(CameraImage image) async {
    try {
      img.Image? convertedImage;
      
      // Handle different camera image formats
      if (image.format.group == ImageFormatGroup.yuv420) {
        // Convert YUV420 to RGB image
        convertedImage = img.Image(
          width: image.width,
          height: image.height,
        );
        
        // Get Y, U, V planes
        final yPlane = image.planes[0];
        final uPlane = image.planes[1];
        final vPlane = image.planes[2];
        
        final yBuffer = yPlane.bytes;
        final uBuffer = uPlane.bytes;
        final vBuffer = vPlane.bytes;
        
        // Use bytesPerRow as stride (correct property name)
        final int yRowStride = yPlane.bytesPerRow;
        final int uvRowStride = uPlane.bytesPerRow;
        final int uvPixelStride = uPlane.bytesPerPixel ?? 1;
        
        // Convert YUV to RGB
        for (int y = 0; y < image.height; y++) {
          for (int x = 0; x < image.width; x++) {
            final int yIndex = y * yRowStride + x;
            final int uvIndex = ((y ~/ 2) * uvRowStride) + (((x ~/ 2) * uvPixelStride)).toInt();
            
            final int yValue = yBuffer[yIndex];
            final int uValue = uBuffer[uvIndex] - 128;
            final int vValue = vBuffer[uvIndex] - 128;
            
            // YUV to RGB conversion formula
            int r = (yValue + (1.402 * vValue).round()).clamp(0, 255);
            int g = (yValue - (0.344136 * uValue).round() - (0.714136 * vValue).round()).clamp(0, 255);
            int b = (yValue + (1.772 * uValue).round()).clamp(0, 255);
            
            convertedImage.setPixelRgba(x, y, r, g, b, 255);
          }
        }
      } else if (image.format.group == ImageFormatGroup.bgra8888) {
        // Already in BGRA format - convert directly
        final bytes = image.planes[0].bytes;
        convertedImage = img.Image.fromBytes(
          width: image.width,
          height: image.height,
          bytes: bytes.buffer,
          order: img.ChannelOrder.bgra,
        );
      } else if (image.format.group == ImageFormatGroup.jpeg) {
        // Already JPEG format - return directly
        return image.planes[0].bytes;
      }
      
      if (convertedImage == null) {
        return Uint8List(0);
      }
      
      // Encode to JPEG
      final jpeg = img.encodeJpg(convertedImage, quality: 75);
      return Uint8List.fromList(jpeg);
      
    } catch (e) {
      if (kDebugMode) {
        print('Error converting image: $e');
      }
      return Uint8List(0);
    }
  }

  Future<void> _stopAnalysis() async {
    try {
      // Stop camera stream
      if (_controller != null) {
        await _controller!.stopImageStream();
      }
      
      // Disconnect WebSocket
      _videoService.disconnect();
      
      setState(() {
        _isRecording = false;
        _isConnected = false;
        _connectionStatus = "Disconnected";
        _statusText = "Analysis stopped";
        _isAnalyzing = false;
      });
      
    } catch (e) {
      setState(() {
        _statusText = "Error stopping analysis: $e";
      });
    }
  }
  
  Future<void> _connectToServer() async {
    if (_serverUrl.isEmpty) {
      setState(() {
        _connectionError = "Please enter a server URL";
      });
      return;
    }
    
    setState(() {
      _connectionStatus = "Connecting...";
      _connectionError = null;
    });
    
    try {
      await _videoService.connect(_serverUrl, port: 8000);
      
      if (_videoService.isConnected) {
        setState(() {
          _statusText = "Connected! Press Start to begin analysis";
        });
      }
    } catch (e) {
      setState(() {
        _connectionError = "Failed to connect: $e";
        _connectionStatus = "Connection failed";
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
    _frameCaptureTimer?.cancel();
    _controller?.dispose();
    _videoService.disconnect();
    _serverUrlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return _buildLoadingScreen();
    }

    return Scaffold(
      body: Stack(
        children: [
          // Camera Preview
          SizedBox.expand(
            child: CameraPreview(_controller!),
          ),

          // Connection Status Overlay
          Positioned(
            top: 10,
            left: 10,
            right: 10,
            child: _buildConnectionStatusOverlay(),
          ),

          // AI Analysis Result Overlay
          if (_currentAnalysis != null)
            Positioned(
              top: 80,
              left: 10,
              right: 10,
              child: _buildAnalysisResultOverlay(),
            ),

          // Status Text - Top Section
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
                      fontSize: 16,
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
                      "Duration: ${_formatDuration(_recordingDuration)} | Frames: $_framesSent",
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),

          // Recording Indicator
          if (_isRecording)
            Positioned(
              top: 50,
              right: 20,
              child: _buildRecordingIndicator(),
            ),

          // Control Buttons - Bottom Section
          Positioned(
            bottom: 30,
            left: 0,
            right: 0,
            child: _buildControlButtons(),
          ),
          
          // Error Dialog
          if (_connectionError != null)
            _buildErrorDialog(),
        ],
      ),
    );
  }
  
  Widget _buildLoadingScreen() {
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
  
  Widget _buildConnectionStatusOverlay() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: _isConnected ? Colors.green.withOpacity(0.8) : Colors.orange.withOpacity(0.8),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            _isConnected ? Icons.cloud_done : Icons.cloud_off,
            color: Colors.white,
            size: 16,
          ),
          const SizedBox(width: 6),
          Text(
            _connectionStatus,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildAnalysisResultOverlay() {
    if (_currentAnalysis == null) return const SizedBox.shrink();
    
    final analysis = _currentAnalysis!;
    Color statusColor;
    
    switch (analysis.drowsinessLevel) {
      case 'awake':
        statusColor = Colors.green;
        break;
      case 'mildly drowsy':
        statusColor = Colors.yellow;
        break;
      case 'moderately drowsy':
        statusColor = Colors.orange;
        break;
      case 'highly drowsy':
        statusColor = Colors.red;
        break;
      default:
        statusColor = Colors.grey;
    }
    
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.7),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: statusColor, width: 2),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                analysis.isDrowsy ? Icons.warning : Icons.check_circle,
                color: statusColor,
                size: 24,
              ),
              const SizedBox(width: 8),
              Text(
                analysis.drowsinessStatus,
                style: TextStyle(
                  color: statusColor,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            "Confidence: ${(analysis.confidence * 100).toStringAsFixed(1)}%",
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 12,
            ),
          ),
          if (analysis.observations.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                analysis.observations.take(2).join(", "),
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                ),
                textAlign: TextAlign.center,
              ),
            ),
        ],
      ),
    );
  }
  
  Widget _buildRecordingIndicator() {
    return Container(
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
            "LIVE",
            style: TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildControlButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        // Server Connection Button
        FloatingActionButton.small(
          onPressed: _showServerDialog,
          backgroundColor: _isConnected ? Colors.green : Colors.white.withOpacity(0.3),
          child: Icon(
            _isConnected ? Icons.link : Icons.link_off,
            color: Colors.white,
          ),
        ),

        // Start/Stop Button
        FloatingActionButton(
          onPressed: _isRecording ? _stopAnalysis : _startAnalysis,
          backgroundColor: _isRecording ? Colors.red : Colors.white,
          foregroundColor: _isRecording ? Colors.white : Colors.red,
          child: Icon(
            _isRecording ? Icons.stop : Icons.play_arrow,
            size: 32,
          ),
        ),

        // Settings Button
        FloatingActionButton.small(
          onPressed: _showServerDialog,
          backgroundColor: Colors.white.withOpacity(0.3),
          child: const Icon(Icons.settings, color: Colors.white),
        ),
      ],
    );
  }
  
  void _showServerDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Server Configuration"),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _serverUrlController,
              decoration: const InputDecoration(
                labelText: "Server IP Address",
                hintText: "localhost or 192.168.1.x",
                prefixIcon: Icon(Icons.dns),
              ),
              onChanged: (value) => _serverUrl = value,
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context);
                      _connectToServer();
                    },
                    child: const Text("Connect"),
                  ),
                ),
              ],
            ),
            if (_connectionError != null) ...[
              const SizedBox(height: 12),
              Text(
                _connectionError!,
                style: const TextStyle(
                  color: Colors.red,
                  fontSize: 12,
                ),
                textAlign: TextAlign.center,
              ),
            ],
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
  }
  
  Widget _buildErrorDialog() {
    return AlertDialog(
      title: const Text("Connection Error"),
      content: Text(_connectionError ?? "An unknown error occurred"),
      actions: [
        TextButton(
          onPressed: () {
            setState(() {
              _connectionError = null;
            });
          },
          child: const Text("Dismiss"),
        ),
        TextButton(
          onPressed: () {
            setState(() {
              _connectionError = null;
            });
            _connectToServer();
          },
          child: const Text("Retry"),
        ),
      ],
    );
  }
}
