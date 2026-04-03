import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:provider/provider.dart';
import 'app_state.dart';
import 'api_service.dart';

class ChatMessage {
  final String id;
  final String role;
  String content;
  List<Citation>? citations;
  Map<String, dynamic>? metadata;
  bool isStreaming;

  ChatMessage({
    required this.id,
    required this.role,
    required this.content,
    this.citations,
    this.metadata,
    this.isStreaming = false,
  });
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<ChatMessage> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final FocusNode _focusNode = FocusNode();
  bool _isLoading = false;
  String? _currentTenant;

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  String _generateId() =>
      DateTime.now().millisecondsSinceEpoch.toRadixString(36);

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _isLoading) return;

    final appState = context.read<AppState>();
    final tenantId = appState.tenantId;

    if (_currentTenant != null && _currentTenant != tenantId) {
      setState(() => _messages.clear());
    }
    _currentTenant = tenantId;

    final userMsg = ChatMessage(
      id: _generateId(),
      role: 'user',
      content: text,
    );
    setState(() {
      _messages.add(userMsg);
      _isLoading = true;
    });
    _controller.clear();
    _scrollToBottom();

    final assistantId = _generateId();
    final api = ApiService(baseUrl: appState.serverUrl);

    final history = _messages
        .where((m) => m.role == 'user' || m.role == 'assistant')
        .take(6)
        .map((m) => {'role': m.role, 'content': m.content})
        .toList();

    try {
      final assistantMsg = ChatMessage(
        id: assistantId,
        role: 'assistant',
        content: '',
        isStreaming: true,
      );
      setState(() => _messages.add(assistantMsg));
      _scrollToBottom();

      String fullContent = '';

      await for (final token in api.chatStream(
        tenantId: tenantId,
        message: text,
        conversationHistory: history,
        useRag: appState.useRag,
      )) {
        fullContent += token;
        setState(() {
          final idx = _messages.indexWhere((m) => m.id == assistantId);
          if (idx != -1) {
            _messages[idx].content = fullContent;
          }
        });
        _scrollToBottom();
      }

      setState(() {
        final idx = _messages.indexWhere((m) => m.id == assistantId);
        if (idx != -1) {
          _messages[idx].isStreaming = false;
        }
      });
    } catch (streamError) {
      setState(() {
        _messages.removeWhere((m) => m.id == assistantId);
      });

      try {
        final response = await api.chat(
          tenantId: tenantId,
          message: text,
          conversationHistory: history,
          useRag: appState.useRag,
        );

        setState(() {
          _messages.add(ChatMessage(
            id: assistantId,
            role: 'assistant',
            content: response.message,
            citations: response.citations,
            metadata: {
              'request_id': response.requestId,
              'model_type': response.modelType,
              'latency_ms': response.latencyMs,
              'grounding_score': response.groundingScore,
              'retrieval_method': response.retrievalMethod,
            },
          ));
        });
      } catch (chatError) {
        setState(() {
          _messages.add(ChatMessage(
            id: assistantId,
            role: 'assistant',
            content:
                '⚠️ Connection error: ${chatError.toString().replaceAll('Exception: ', '')}\n\nMake sure the server is running.',
          ));
        });
      }
    }

    setState(() => _isLoading = false);
    _scrollToBottom();
  }

  void _submitFeedback(ChatMessage message, int rating) async {
    final appState = context.read<AppState>();
    final requestId = message.metadata?['request_id'];
    if (requestId == null) return;

    try {
      final api = ApiService(baseUrl: appState.serverUrl);
      await api.submitFeedback(
        requestId: requestId,
        tenantId: appState.tenantId,
        rating: rating,
      );

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(rating >= 4 ? '👍 Thanks!' : '👎 Thanks for the feedback'),
            duration: const Duration(seconds: 1),
          ),
        );
      }
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();

    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Text(appState.tenantEmoji, style: const TextStyle(fontSize: 24)),
            const SizedBox(width: 8),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${appState.tenantId.toUpperCase()} Assistant',
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
                Text(
                  appState.isConnected ? 'Connected' : 'Disconnected',
                  style: TextStyle(
                    fontSize: 11,
                    color: appState.isConnected ? Colors.green : Colors.red,
                  ),
                ),
              ],
            ),
          ],
        ),
        actions: [
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'sis', label: Text('SIS', style: TextStyle(fontSize: 12))),
              ButtonSegment(value: 'mfg', label: Text('MFG', style: TextStyle(fontSize: 12))),
            ],
            selected: {appState.tenantId},
            onSelectionChanged: (val) {
              appState.setTenant(val.first);
              setState(() => _messages.clear());
            },
            style: ButtonStyle(
              visualDensity: VisualDensity.compact,
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
            ),
          ),
          const SizedBox(width: 8),
          if (_messages.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_outline, size: 20),
              onPressed: () => setState(() => _messages.clear()),
              tooltip: 'Clear chat',
            ),
        ],
      ),
      body: Column(
        children: [
          if (!appState.isConnected)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              color: Colors.orange.shade50,
              child: Row(
                children: [
                  Icon(Icons.wifi_off, size: 16, color: Colors.orange.shade700),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Server not reachable. Check Settings.',
                      style: TextStyle(fontSize: 12, color: Colors.orange.shade800),
                    ),
                  ),
                  TextButton(
                    onPressed: () => appState.checkConnection(),
                    child: const Text('Retry', style: TextStyle(fontSize: 12)),
                  ),
                ],
              ),
            ),

          Expanded(
            child: _messages.isEmpty
                ? _buildEmptyState(appState)
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    itemCount: _messages.length,
                    itemBuilder: (context, index) =>
                        _buildMessageBubble(_messages[index], appState),
                  ),
          ),

          _buildInputArea(appState),
        ],
      ),
    );
  }

  Widget _buildEmptyState(AppState appState) {
    final sampleQuestions = appState.tenantId == 'sis'
        ? [
            'What documents are needed for enrollment?',
            'How does FERPA protect student records?',
            'What happens after 5 unexcused absences?',
            'What is the grade change procedure?',
          ]
        : [
            'What is the assembly line startup procedure?',
            'How are critical defects handled?',
            'What is the lockout/tagout process?',
            'What triggers a CAPA?',
          ];

    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(appState.tenantEmoji, style: const TextStyle(fontSize: 48)),
            const SizedBox(height: 12),
            Text(
              '${appState.tenantId.toUpperCase()} Assistant',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            Text(
              appState.tenantId == 'sis'
                  ? 'Ask about enrollment, attendance, grading, FERPA...'
                  : 'Ask about SOPs, quality, safety, maintenance...',
              style: Theme.of(context)
                  .textTheme
                  .bodyMedium
                  ?.copyWith(color: Colors.grey.shade600),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            ...sampleQuestions.map((q) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: OutlinedButton(
                onPressed: () {
                  _controller.text = q;
                  _sendMessage();
                },
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: SizedBox(
                  width: double.infinity,
                  child: Text(q, style: const TextStyle(fontSize: 13)),
                ),
              ),
            )),
          ],
        ),
      ),
    );
  }

  Widget _buildMessageBubble(ChatMessage message, AppState appState) {
    final isUser = message.role == 'user';

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.85,
        ),
        margin: const EdgeInsets.only(bottom: 12),
        child: Column(
          crossAxisAlignment:
              isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
          children: [
            if (!isUser)
              Padding(
                padding: const EdgeInsets.only(left: 4, bottom: 4),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(appState.tenantEmoji, style: const TextStyle(fontSize: 14)),
                    const SizedBox(width: 4),
                    Text(
                      '${appState.tenantId.toUpperCase()} Assistant',
                      style: TextStyle(fontSize: 11, color: Colors.grey.shade500),
                    ),
                  ],
                ),
              ),

            Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: isUser ? appState.tenantColor : Colors.white,
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(16),
                  topRight: const Radius.circular(16),
                  bottomLeft: Radius.circular(isUser ? 16 : 4),
                  bottomRight: Radius.circular(isUser ? 4 : 16),
                ),
                border: isUser ? null : Border.all(color: Colors.grey.shade200),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (isUser)
                    Text(
                      message.content,
                      style: const TextStyle(color: Colors.white, fontSize: 14),
                    )
                  else if (message.content.isEmpty && message.isStreaming)
                    SizedBox(
                      width: 40,
                      child: LinearProgressIndicator(
                        color: appState.tenantColor,
                        backgroundColor: Colors.grey.shade200,
                      ),
                    )
                  else
                    MarkdownBody(
                      data: message.content + (message.isStreaming ? ' ▋' : ''),
                      styleSheet: MarkdownStyleSheet(
                        p: const TextStyle(fontSize: 14, height: 1.4),
                        code: TextStyle(
                          fontSize: 12,
                          backgroundColor: Colors.grey.shade100,
                        ),
                      ),
                    ),

                  if (message.citations != null &&
                      message.citations!.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Divider(height: 1, color: Colors.grey.shade200),
                    const SizedBox(height: 8),
                    Text(
                      'Sources',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: Colors.grey.shade600,
                      ),
                    ),
                    const SizedBox(height: 4),
                    ...message.citations!.asMap().entries.map((entry) {
                      final i = entry.key;
                      final c = entry.value;
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 3),
                        child: Row(
                          children: [
                            Container(
                              width: 18,
                              height: 18,
                              decoration: BoxDecoration(
                                color: appState.tenantColor,
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: Center(
                                child: Text(
                                  '${i + 1}',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 10,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 6),
                            Expanded(
                              child: Text(
                                '${c.title} • ${c.topic}',
                                style: TextStyle(
                                  fontSize: 11,
                                  color: Colors.grey.shade700,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            Text(
                              '${(c.relevanceScore * 100).toStringAsFixed(0)}%',
                              style: TextStyle(
                                fontSize: 10,
                                color: Colors.grey.shade500,
                              ),
                            ),
                          ],
                        ),
                      );
                    }),
                  ],

                  if (!isUser &&
                      !message.isStreaming &&
                      message.metadata != null) ...[
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        if (message.metadata!['latency_ms'] != null)
                          _metaChip(
                            '${(message.metadata!['latency_ms'] as num).round()}ms',
                          ),
                        if (message.metadata!['model_type'] != null &&
                            (message.metadata!['model_type'] as String).isNotEmpty)
                          _metaChip(message.metadata!['model_type'] as String),
                        if (message.metadata!['grounding_score'] != null)
                          _metaChip(
                            '${((message.metadata!['grounding_score'] as num) * 100).round()}% grounded',
                          ),
                        const Spacer(),
                        if (message.metadata!['request_id'] != null &&
                            (message.metadata!['request_id'] as String).isNotEmpty) ...[
                          InkWell(
                            onTap: () => _submitFeedback(message, 5),
                            child: const Padding(
                              padding: EdgeInsets.all(4),
                              child: Text('👍', style: TextStyle(fontSize: 16)),
                            ),
                          ),
                          InkWell(
                            onTap: () => _submitFeedback(message, 1),
                            child: const Padding(
                              padding: EdgeInsets.all(4),
                              child: Text('👎', style: TextStyle(fontSize: 16)),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _metaChip(String text) {
    return Container(
      margin: const EdgeInsets.only(right: 6),
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        text,
        style: TextStyle(fontSize: 10, color: Colors.grey.shade600),
      ),
    );
  }

  Widget _buildInputArea(AppState appState) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(top: BorderSide(color: Colors.grey.shade200)),
      ),
      child: SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                FilterChip(
                  label: const Text('RAG', style: TextStyle(fontSize: 11)),
                  selected: appState.useRag,
                  onSelected: (v) => appState.setUseRag(v),
                  materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  visualDensity: VisualDensity.compact,
                  selectedColor: appState.tenantColor.withOpacity(0.15),
                ),
                const Spacer(),
                if (!appState.isConnected)
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.cloud_off, size: 14, color: Colors.red.shade300),
                      const SizedBox(width: 4),
                      Text(
                        'Offline',
                        style: TextStyle(fontSize: 11, color: Colors.red.shade300),
                      ),
                    ],
                  ),
              ],
            ),
            const SizedBox(height: 6),
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    focusNode: _focusNode,
                    maxLines: 4,
                    minLines: 1,
                    textInputAction: TextInputAction.send,
                    onSubmitted: (_) => _sendMessage(),
                    enabled: !_isLoading,
                    decoration: InputDecoration(
                      hintText: appState.tenantId == 'sis'
                          ? 'Ask about student information...'
                          : 'Ask about manufacturing...',
                      hintStyle: TextStyle(
                        fontSize: 14,
                        color: Colors.grey.shade400,
                      ),
                      filled: true,
                      fillColor: Colors.grey.shade50,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                        borderSide: BorderSide(color: Colors.grey.shade300),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                        borderSide: BorderSide(color: Colors.grey.shade300),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                        borderSide:
                            BorderSide(color: appState.tenantColor, width: 2),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 10,
                      ),
                      isDense: true,
                    ),
                    style: const TextStyle(fontSize: 14),
                  ),
                ),
                const SizedBox(width: 8),
                FloatingActionButton.small(
                  onPressed: _isLoading ? null : _sendMessage,
                  backgroundColor:
                      _isLoading ? Colors.grey.shade300 : appState.tenantColor,
                  child: _isLoading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.send, size: 18, color: Colors.white),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
