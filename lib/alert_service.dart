import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'package:web/web.dart' as web;
import 'package:flutter/foundation.dart';

class AlertService {
  static int _drowsyFrameCount = 0;
  static bool _isAlerting = false;
  static const int _triggerFrames = 10;

  static void onDrowsinessLevel(String level) {
    if (level == 'awake') {
      _drowsyFrameCount = 0;
      if (_isAlerting) {
        _stopAlarm();
        _isAlerting = false;
      }
      return;
    }

    _drowsyFrameCount++;

    if (_drowsyFrameCount >= _triggerFrames) {
      if (!_isAlerting) {
        _isAlerting = true;
        _startAlarm(level);
      } else {
        _updateAlarmSpeed();
      }
    }
  }

  static void _startAlarm(String level) {
    if (!kIsWeb) return;
    try {
      final freq = _getFrequency(level);
      final js = '''
        window._alarmFreq = $freq;
        window._alarmInterval_ms = 300;
        if (window._alarmCtx) window._alarmCtx.close();
        window._alarmCtx = new (window.AudioContext || window.webkitAudioContext)();
        window._alarmBeep = function() {
          var ctx = window._alarmCtx;
          if (!ctx) return;
          var osc1 = ctx.createOscillator();
          var osc2 = ctx.createOscillator();
          var gain = ctx.createGain();
          osc1.connect(gain); osc2.connect(gain);
          gain.connect(ctx.destination);
          osc1.type = 'sawtooth'; osc2.type = 'square';
          osc1.frequency.value = window._alarmFreq;
          osc2.frequency.value = window._alarmFreq * 1.02;
          gain.gain.setValueAtTime(0, ctx.currentTime);
          gain.gain.linearRampToValueAtTime(0.7, ctx.currentTime + 0.03);
          gain.gain.setValueAtTime(0.7, ctx.currentTime + 0.12);
          gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.18);
          osc1.start(ctx.currentTime); osc2.start(ctx.currentTime);
          osc1.stop(ctx.currentTime + 0.18); osc2.stop(ctx.currentTime + 0.18);
        };
        if (window._alarmTimer) clearInterval(window._alarmTimer);
        window._alarmTimer = setInterval(window._alarmBeep, window._alarmInterval_ms);
        window._alarmBeep();
      '''.toJS;
      web.window.callMethod('eval'.toJS, js);
      if (kDebugMode) print('[ALERT] Alarm started: $level @ ${_getFrequency(level)}Hz');
    } catch (e) {
      if (kDebugMode) print('[ALERT] Alarm error: $e');
    }
  }

  static void _updateAlarmSpeed() {
    if (!kIsWeb) return;
    try {
      final interval = _drowsyFrameCount > 50 ? 120 : _drowsyFrameCount > 25 ? 200 : 300;
      final js = '''
        if (window._alarmTimer) clearInterval(window._alarmTimer);
        window._alarmInterval_ms = $interval;
        window._alarmTimer = setInterval(window._alarmBeep, $interval);
      '''.toJS;
      web.window.callMethod('eval'.toJS, js);
    } catch (e) {
      if (kDebugMode) print('[ALERT] Update speed error: $e');
    }
  }

  static void _stopAlarm() {
    if (!kIsWeb) return;
    try {
      final js = '''
        if (window._alarmTimer) { clearInterval(window._alarmTimer); window._alarmTimer = null; }
        if (window._alarmCtx) { window._alarmCtx.close(); window._alarmCtx = null; }
      '''.toJS;
      web.window.callMethod('eval'.toJS, js);
      if (kDebugMode) print('[ALERT] Alarm stopped');
    } catch (e) {
      if (kDebugMode) print('[ALERT] Stop error: $e');
    }
  }

  static int _getFrequency(String level) {
    switch (level) {
      case 'highly drowsy': return 1200;
      case 'moderately drowsy': return 900;
      case 'mildly drowsy': return 700;
      default: return 440;
    }
  }

  static void reset() {
    _drowsyFrameCount = 0;
    _stopAlarm();
    _isAlerting = false;
  }
}
