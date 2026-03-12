import 'package:flutter/material.dart';

class FaceLandmark {
  final double x;
  final double y;
  FaceLandmark(this.x, this.y);
}

class FaceMeshPainter extends CustomPainter {
  final List<FaceLandmark> landmarks;
  final String drowsinessLevel;

  FaceMeshPainter({required this.landmarks, required this.drowsinessLevel});

  Color get meshColor {
    switch (drowsinessLevel) {
      case 'awake':
        return Colors.greenAccent;
      case 'mildly drowsy':
        return Colors.yellowAccent;
      case 'moderately drowsy':
        return Colors.orangeAccent;
      case 'highly drowsy':
        return Colors.redAccent;
      default:
        return Colors.blueAccent;
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.isEmpty) return;

    final dotPaint = Paint()
      ..color = meshColor.withOpacity(0.85)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.fill;

    final linePaint = Paint()
      ..color = meshColor.withOpacity(0.35)
      ..strokeWidth = 0.8
      ..style = PaintingStyle.stroke;

    // Draw landmark dots
    for (final lm in landmarks) {
      canvas.drawCircle(
        Offset(lm.x * size.width, lm.y * size.height),
        1.8,
        dotPaint,
      );
    }

    // Draw eye contours
    _drawContour(canvas, linePaint, size, _leftEye);
    _drawContour(canvas, linePaint, size, _rightEye);
    _drawContour(canvas, linePaint, size, _lips);
    _drawContour(canvas, linePaint, size, _faceOval);
  }

  // Landmark index groups
  static const _leftEye = [33, 160, 158, 133, 153, 144, 33];
  static const _rightEye = [362, 385, 387, 263, 373, 380, 362];
  static const _lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61];
  static const _faceOval = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
    10,
  ];

  void _drawContour(Canvas canvas, Paint paint, Size size, List<int> indices) {
    if (landmarks.isEmpty) return;
    final path = Path();
    bool first = true;
    for (final idx in indices) {
      if (idx >= landmarks.length) continue;
      final lm = landmarks[idx];
      final x = lm.x * size.width;
      final y = lm.y * size.height;
      if (first) {
        path.moveTo(x, y);
        first = false;
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(FaceMeshPainter oldDelegate) =>
      oldDelegate.landmarks != landmarks ||
      oldDelegate.drowsinessLevel != drowsinessLevel;
}
