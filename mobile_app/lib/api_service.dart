import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

class Citation {
  final String title;
  final String topic;
  final String sourceFile;
  final double relevanceScore;
  final String citationKey;

  Citation({
    required this.title,
    required this.topic,
    required this.sourceFile,
    required this.relevanceScore,
    required this.citationKey,
  });

  factory Citation.fromJson(Map<String, dynamic> json) {
    return Citation(
      title: json['title'] ?? '',
      topic: json['topic'] ?? '',
      sourceFile: json['source_file'] ?? '',
      relevanceScore: (json['relevance_score'] ?? 0).toDouble(),
      citationKey: json['citation_key'] ?? '',
    );
  }
}

class ChatResponseData {
  final String message;
  final List<Citation> citations;
  final String modelType;
  final String requestId;
  final double? groundingScore;
  final double latencyMs;
  final String retrievalMethod;
  final bool isCanary;

  ChatResponseData({
    required this.message,
    required this.citations,
    this.modelType = '',
    this.requestId = '',
    this.groundingScore,
    this.latencyMs = 0,
    this.retrievalMethod = '',
    this.isCanary = false,
  });

  factory ChatResponseData.fromJson(Map<String, dynamic> json) {
    return ChatResponseData(
      message: json['message'] ?? '',
      citations: (json['citations'] as List?)
              ?.map((c) => Citation.fromJson(c))
              .toList() ??
          [],
      modelType: json['model_type'] ?? '',
      requestId: json['request_id'] ?? '',
      groundingScore: json['grounding_score']?.toDouble(),
      latencyMs: (json['latency_ms'] ?? 0).toDouble(),
      retrievalMethod: json['retrieval_method'] ?? '',
      isCanary: json['is_canary'] ?? false,
    );
  }
}

class ApiService {
  final String baseUrl;
  final Duration timeout;

  ApiService({
    required this.baseUrl,
    this.timeout = const Duration(seconds: 60),
  });

  Future<Map<String, dynamic>> healthCheck() async {
    final response = await http
        .get(Uri.parse('$baseUrl/health'))
        .timeout(const Duration(seconds: 5));
    if (response.statusCode == 200) {
      return json.decode(response.body);
    }
    throw Exception('Health check failed: ${response.statusCode}');
  }

  Future<ChatResponseData> chat({
    required String tenantId,
    required String message,
    List<Map<String, String>> conversationHistory = const [],
    bool useRag = true,
    int maxNewTokens = 512,
    double temperature = 0.7,
  }) async {
    final body = json.encode({
      'tenant_id': tenantId,
      'message': message,
      'conversation_history': conversationHistory,
      'use_rag': useRag,
      'use_streaming': false,
      'max_new_tokens': maxNewTokens,
      'temperature': temperature,
    });

    final response = await http
        .post(
          Uri.parse('$baseUrl/chat'),
          headers: {'Content-Type': 'application/json'},
          body: body,
        )
        .timeout(timeout);

    if (response.statusCode == 200) {
      return ChatResponseData.fromJson(json.decode(response.body));
    }
    throw Exception('Chat failed: ${response.statusCode} — ${response.body}');
  }

  Stream<String> chatStream({
    required String tenantId,
    required String message,
    List<Map<String, String>> conversationHistory = const [],
    bool useRag = true,
    int maxNewTokens = 512,
    double temperature = 0.7,
  }) async* {
    final request = http.Request('POST', Uri.parse('$baseUrl/chat/stream'));
    request.headers['Content-Type'] = 'application/json';
    request.headers['Accept'] = 'text/event-stream';
    request.body = json.encode({
      'tenant_id': tenantId,
      'message': message,
      'conversation_history': conversationHistory,
      'use_rag': useRag,
      'use_streaming': true,
      'max_new_tokens': maxNewTokens,
      'temperature': temperature,
    });

    final client = http.Client();
    try {
      final streamedResponse = await client.send(request).timeout(timeout);

      if (streamedResponse.statusCode != 200) {
        throw Exception('Stream failed: ${streamedResponse.statusCode}');
      }

      String buffer = '';
      await for (final chunk
          in streamedResponse.stream.transform(utf8.decoder)) {
        buffer += chunk;
        final lines = buffer.split('\n');
        buffer = lines.removeLast();

        for (final line in lines) {
          if (line.startsWith('data:')) {
            final dataStr = line.substring(5).trim();
            if (dataStr.isEmpty) continue;

            try {
              final data = json.decode(dataStr);
              if (data['token'] != null) {
                yield data['token'] as String;
              }
              if (data['error'] != null) {
                yield '\n⚠️ Error: ${data['error']}';
              }
            } catch (_) {
              // Skip malformed JSON
            }
          }
        }
      }
    } finally {
      client.close();
    }
  }

  Future<void> submitFeedback({
    required String requestId,
    required String tenantId,
    required int rating,
    String? comment,
  }) async {
    await http
        .post(
          Uri.parse('$baseUrl/feedback'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({
            'request_id': requestId,
            'tenant_id': tenantId,
            'rating': rating,
            'feedback_type': 'thumbs',
            'comment': comment,
          }),
        )
        .timeout(const Duration(seconds: 10));
  }

  Future<Map<String, dynamic>> getStats({String? tenantId}) async {
    final uri = tenantId != null
        ? Uri.parse('$baseUrl/stats?tenant_id=$tenantId')
        : Uri.parse('$baseUrl/stats');
    final response = await http.get(uri).timeout(const Duration(seconds: 10));
    if (response.statusCode == 200) {
      return json.decode(response.body);
    }
    throw Exception('Stats failed: ${response.statusCode}');
  }
}
