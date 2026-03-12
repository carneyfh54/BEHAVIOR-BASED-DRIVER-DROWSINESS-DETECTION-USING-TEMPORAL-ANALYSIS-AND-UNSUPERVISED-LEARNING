import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as ws_status;
import 'face_mesh_painter.dart';

class VideoAnalysisService {
  static final VideoAnalysisService _instance = VideoAnalysisService._internal();
  factory VideoAnalysisService() => _instance;
  VideoAnalysisService._internal();

  WebSocketChannel? _channel;
  String? _serverUrl;
  bool _isConnected = false;
  int _frameCount = 0;
  int _analysisInterval = 1;

  Function(String type, dynamic data)? _onMessage;
  Function()? _onConnected;
  Function(String error)? _onError;
  Function()? _onDisconnected;

  void setCallbacks({
    Function(String type, dynamic data)? onMessage,
    Function()? onConnected,
    Function(String error)? onError,
    Function()? onDisconnected,
  }) {
    _onMessage = onMessage;
    _onConnected = onConnected;
    _onError = onError;
    _onDisconnected = onDisconnected;
  }

  Future<void> connect(String serverUrl, {int port = 8000}) async {
    _serverUrl = serverUrl;
    try {
      String wsUrl;
      if (serverUrl.startsWith('ws://') || serverUrl.startsWith('wss://')) {
        wsUrl = '$serverUrl/ws/video-analysis/';
      } else {
        wsUrl = 'ws://$serverUrl:$port/ws/video-analysis/';
      }
      if (kDebugMode) print('Connecting to: $wsUrl');

      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      _channel!.stream.listen(
        (message) => _handleMessage(message),
        onDone: () {
          _isConnected = false;
          _onDisconnected?.call();
        },
        onError: (error) {
          _isConnected = false;
          _onError?.call(error.toString());
        },
        cancelOnError: false,
      );

      await Future.delayed(const Duration(milliseconds: 500));
      _isConnected = true;
      _onConnected?.call();
      if (kDebugMode) print('WebSocket connected successfully');
    } catch (e) {
      _isConnected = false;
      _onError?.call('Connection failed: $e');
      rethrow;
    }
  }

  void _handleMessage(dynamic message) {
    try {
      if (message is String) {
        final data = jsonDecode(message);
        final type = data['type'] ?? 'unknown';
        if (kDebugMode) print('Received: $type');
        _onMessage?.call(type, data);
      }
    } catch (e) {
      if (kDebugMode) print('Error parsing message: $e');
    }
  }

  Future<void> sendFrame(Uint8List frameData) async {
    if (!_isConnected || _channel == null) return;
    try {
      _frameCount++;
      if (_frameCount % _analysisInterval == 0) {
        _channel!.sink.add(frameData);
      }
    } catch (e) {
      _onError?.call('Failed to send frame: $e');
    }
  }

  void disconnect() {
    _channel?.sink.close(ws_status.normalClosure);
    _channel = null;
    _isConnected = false;
    _onDisconnected?.call();
  }

  bool get isConnected => _isConnected;
  int get frameCount => _frameCount;
  String? get serverUrl => _serverUrl;

  static AnalysisResult parseAnalysisResult(dynamic data) {
    if (data['type'] != 'analysis_result') {
      throw ArgumentError('Invalid message type');
    }
    final d = data['data'];
    return AnalysisResult(
      drowsinessLevel: d['drowsiness_level'] ?? 'unknown',
      confidence: (d['confidence'] ?? 0.0).toDouble(),
      observations: List<String>.from(d['observations'] ?? []),
      recommendedAction: d['recommended_action'] ?? '',
      frameNumber: data['frame_number'] ?? 0,
      landmarks: (data['landmarks'] as List<dynamic>? ?? [])
          .map((l) => FaceLandmark(
                (l['x'] as num).toDouble(),
                (l['y'] as num).toDouble(),
              ))
          .toList(),
    );
  }
}

class AnalysisResult {
  final String drowsinessLevel;
  final double confidence;
  final List<String> observations;
  final String recommendedAction;
  final int frameNumber;
  final List<FaceLandmark> landmarks;

  AnalysisResult({
    required this.drowsinessLevel,
    required this.confidence,
    required this.observations,
    required this.recommendedAction,
    required this.frameNumber,
    this.landmarks = const [],
  });

  bool get isDrowsy =>
      drowsinessLevel == 'mildly drowsy' ||
      drowsinessLevel == 'moderately drowsy' ||
      drowsinessLevel == 'highly drowsy';

  bool get isHighlyDrowsy => drowsinessLevel == 'highly drowsy';

  int get drowsinessLevelNumeric {
    switch (drowsinessLevel) {
      case 'awake': return 0;
      case 'mildly drowsy': return 1;
      case 'moderately drowsy': return 2;
      case 'highly drowsy': return 3;
      default: return -1;
    }
  }

  String get drowsinessStatus {
    switch (drowsinessLevel) {
      case 'awake': return '🟢 AWAKE';
      case 'mildly drowsy': return '🟡 MILDLY DROWSY';
      case 'moderately drowsy': return '🟠 MODERATELY DROWSY';
      case 'highly drowsy': return '🔴 HIGHLY DROWSY';
      default: return '⚪ UNKNOWN';
    }
  }
}
