/// WebSocket Service for Real-time Video Analysis
/// Handles connection to Django backend and video frame streaming
library;
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as ws_status;

class VideoAnalysisService {
  static final VideoAnalysisService _instance = VideoAnalysisService._internal();
  
  factory VideoAnalysisService() {
    return _instance;
  }
  
  VideoAnalysisService._internal();
  
  WebSocketChannel? _channel;
  String? _serverUrl;
  bool _isConnected = false;
  int _frameCount = 0;
  
  // Callbacks
  Function(String type, dynamic data)? _onMessage;
  Function()? _onConnected;
  Function(String error)? _onError;
  Function()? _onDisconnected;
  
  // Analysis configuration
  int _analysisInterval = 1; // Analyze every Nth frame
  bool _isStreaming = false;
  
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
      // Handle different URL formats
      String wsUrl;
      if (serverUrl.startsWith('ws://') || serverUrl.startsWith('wss://')) {
        wsUrl = '$serverUrl/ws/video-analysis/';
      } else {
        wsUrl = 'ws://$serverUrl/ws/video-analysis/';
      }
      
      if (kDebugMode) {
        print('Connecting to WebSocket: $wsUrl');
      }
      
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      
      // Listen for stream errors
      _channel!.stream.handleError((error) {
        if (kDebugMode) {
          print('WebSocket error: $error');
        }
        _isConnected = false;
        _onError?.call(error.toString());
      });
      
      // Listen for connection close
      _channel!.stream.listen(
        (message) => _handleMessage(message),
        onDone: () {
          if (kDebugMode) {
            print('WebSocket connection closed');
          }
          _isConnected = false;
          _onDisconnected?.call();
        },
        onError: (error) {
          if (kDebugMode) {
            print('WebSocket error: $error');
          }
          _isConnected = false;
          _onError?.call(error.toString());
        },
        cancelOnError: false,
      );
      
      // Wait a moment for connection to establish
      await Future.delayed(const Duration(milliseconds: 500));
      _isConnected = true;
      _onConnected?.call();
      
      if (kDebugMode) {
        print('WebSocket connected successfully');
      }
      
    } catch (e) {
      if (kDebugMode) {
        print('Failed to connect to WebSocket: $e');
      }
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
        
        if (kDebugMode) {
          print('Received message type: $type');
        }
        
        _onMessage?.call(type, data);
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error parsing message: $e');
      }
    }
  }
  
  Future<void> sendFrame(Uint8List frameData) async {
    if (!_isConnected || _channel == null) {
      if (kDebugMode) {
        print('Cannot send frame: Not connected');
      }
      return;
    }
    
    try {
      _frameCount++;
      
      // Only send every Nth frame based on analysis interval
      if (_frameCount % _analysisInterval == 0) {
        _channel!.sink.add(frameData);
        
        if (kDebugMode) {
          print('Frame sent: $_frameCount (size: ${frameData.length} bytes)');
        }
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error sending frame: $e');
      }
      _onError?.call('Failed to send frame: $e');
    }
  }
  
  Future<void> sendConfiguration({int interval = 1}) async {
    if (!_isConnected || _channel == null) {
      return;
    }
    
    _analysisInterval = interval;
    
    final config = jsonEncode({
      'type': 'configure',
      'interval': interval,
    });
    
    _channel!.sink.add(config);
    
    if (kDebugMode) {
      print('Configuration sent: interval=$interval');
    }
  }
  
  void setAnalysisInterval(int interval) {
    _analysisInterval = interval;
    sendConfiguration(interval: interval);
  }
  
  void disconnect() {
    _isStreaming = false;
    
    if (_channel != null) {
      _channel!.sink.close(ws_status.goingAway);
      _channel = null;
    }
    
    _isConnected = false;
    
    if (kDebugMode) {
      print('WebSocket disconnected');
    }
    
    _onDisconnected?.call();
  }
  
  bool get isConnected => _isConnected;
  int get frameCount => _frameCount;
  String? get serverUrl => _serverUrl;
  
  // Analysis result parsing
  static AnalysisResult parseAnalysisResult(dynamic data) {
    if (data['type'] != 'analysis_result') {
      throw ArgumentError('Invalid message type');
    }
    
    final analysisData = data['data'];
    
    return AnalysisResult(
      drowsinessLevel: analysisData['drowsiness_level'] ?? 'unknown',
      confidence: (analysisData['confidence'] ?? 0.0).toDouble(),
      observations: List<String>.from(analysisData['observations'] ?? []),
      recommendedAction: analysisData['recommended_action'] ?? '',
      frameNumber: data['frame_number'] ?? 0,
      hasError: analysisData.containsKey('error'),
    );
  }
}

class AnalysisResult {
  final String drowsinessLevel;
  final double confidence;
  final List<String> observations;
  final String recommendedAction;
  final int frameNumber;
  final bool hasError;
  
  AnalysisResult({
    required this.drowsinessLevel,
    required this.confidence,
    required this.observations,
    required this.recommendedAction,
    required this.frameNumber,
    this.hasError = false,
  });
  
  bool get isDrowsy => 
      drowsinessLevel == 'mildly drowsy' || 
      drowsinessLevel == 'moderately drowsy' || 
      drowsinessLevel == 'highly drowsy';
  
  bool get isHighlyDrowsy => drowsinessLevel == 'highly drowsy';
  
  // Drowsiness level as numeric value for display
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
