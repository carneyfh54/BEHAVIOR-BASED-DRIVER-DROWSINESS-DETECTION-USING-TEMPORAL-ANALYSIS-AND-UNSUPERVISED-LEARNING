import 'dart:typed_data';

// ignore: avoid_web_libraries_in_flutter
import 'dart:js_interop';
import 'package:web/web.dart' as web;

class WebCameraService {
  web.HTMLVideoElement? _videoElement;
  web.HTMLCanvasElement? _canvas;

  Future<void> initialize() async {
    final videos = web.document.querySelectorAll('video');
    if (videos.length > 0) {
      _videoElement = videos.item(0) as web.HTMLVideoElement;
    }
    _canvas = web.document.createElement('canvas') as web.HTMLCanvasElement;
  }

  Future<Uint8List?> captureFrame() async {
    if (_videoElement == null) await initialize();
    if (_videoElement == null) return null;

    final video = _videoElement!;
    final w = video.videoWidth;
    final h = video.videoHeight;
    if (w == 0 || h == 0) return null;

    _canvas!.width = w;
    _canvas!.height = h;

    final ctx = _canvas!.getContext('2d') as web.CanvasRenderingContext2D;
    ctx.drawImage(video, 0, 0);

    final dataUrl = _canvas!.toDataURL('image/jpeg', 0.7.toJS);
    final base64 = dataUrl.split(',').last;
    return _base64Decode(base64);
  }

  Uint8List _base64Decode(String source) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    final bytes = <int>[];
    int i = 0;
    final src = source.replaceAll(RegExp(r'[^A-Za-z0-9+/]'), '');
    while (i < src.length) {
      final a = chars.indexOf(src[i++]);
      final b = i < src.length ? chars.indexOf(src[i++]) : 0;
      final c = i < src.length ? chars.indexOf(src[i++]) : 0;
      final d = i < src.length ? chars.indexOf(src[i++]) : 0;
      bytes.add((a << 2) | (b >> 4));
      if (c != 64) bytes.add(((b & 0xf) << 4) | (c >> 2));
      if (d != 64) bytes.add(((c & 0x3) << 6) | d);
    }
    return Uint8List.fromList(bytes);
  }
}
