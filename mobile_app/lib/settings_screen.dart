import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'app_state.dart';
import 'api_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _urlController = TextEditingController();
  Map<String, dynamic>? _healthData;
  Map<String, dynamic>? _statsData;
  bool _isChecking = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    final appState = context.read<AppState>();
    _urlController.text = appState.serverUrl;
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _checkServer() async {
    setState(() {
      _isChecking = true;
      _error = null;
      _healthData = null;
      _statsData = null;
    });

    final appState = context.read<AppState>();
    final api = ApiService(baseUrl: appState.serverUrl);

    try {
      final health = await api.healthCheck();
      final stats = await api.getStats(tenantId: appState.tenantId);

      setState(() {
        _healthData = health;
        _statsData = stats;
      });

      appState.checkConnection();
    } catch (e) {
      setState(() {
        _error = e.toString().replaceAll('Exception: ', '');
      });
    }

    setState(() => _isChecking = false);
  }

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _sectionHeader('Server Connection'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  TextField(
                    controller: _urlController,
                    decoration: const InputDecoration(
                      labelText: 'Server URL',
                      hintText: 'http://10.0.2.2:8000',
                      border: OutlineInputBorder(),
                      helperText: 'Use 10.0.2.2 for Android emulator → host',
                      isDense: true,
                    ),
                    onSubmitted: (val) {
                      appState.setServerUrl(val);
                      _checkServer();
                    },
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      FilledButton.icon(
                        onPressed: _isChecking
                            ? null
                            : () {
                                appState.setServerUrl(_urlController.text);
                                _checkServer();
                              },
                        icon: _isChecking
                            ? const SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              )
                            : const Icon(Icons.refresh, size: 18),
                        label: const Text('Test Connection'),
                      ),
                      const SizedBox(width: 12),
                      Icon(
                        appState.isConnected ? Icons.check_circle : Icons.error,
                        color: appState.isConnected ? Colors.green : Colors.red,
                        size: 20,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        appState.isConnected ? 'Connected' : 'Disconnected',
                        style: TextStyle(
                          color: appState.isConnected ? Colors.green : Colors.red,
                          fontSize: 13,
                        ),
                      ),
                    ],
                  ),
                  if (_error != null) ...[
                    const SizedBox(height: 8),
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.red.shade50,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        _error!,
                        style: TextStyle(fontSize: 12, color: Colors.red.shade700),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          _sectionHeader('Tenant Configuration'),
          Card(
            child: Column(
              children: [
                RadioListTile<String>(
                  title: const Text('SIS — Education'),
                  subtitle: const Text(
                    'Student Information System, FERPA compliance',
                    style: TextStyle(fontSize: 12),
                  ),
                  value: 'sis',
                  groupValue: appState.tenantId,
                  onChanged: (v) => appState.setTenant(v!),
                ),
                const Divider(height: 1),
                RadioListTile<String>(
                  title: const Text('MFG — Manufacturing'),
                  subtitle: const Text(
                    'Quality control, safety protocols, ISO compliance',
                    style: TextStyle(fontSize: 12),
                  ),
                  value: 'mfg',
                  groupValue: appState.tenantId,
                  onChanged: (v) => appState.setTenant(v!),
                ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          _sectionHeader('Chat Options'),
          Card(
            child: Column(
              children: [
                SwitchListTile(
                  title: const Text('RAG Retrieval'),
                  subtitle: const Text(
                    'Use knowledge base for grounded answers',
                    style: TextStyle(fontSize: 12),
                  ),
                  value: appState.useRag,
                  onChanged: (v) => appState.setUseRag(v),
                ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          if (_healthData != null) ...[
            _sectionHeader('Server Status'),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _infoRow('Status', _healthData!['status'] ?? 'unknown'),
                    _infoRow(
                      'GPU',
                      _healthData!['gpu_available'] == true
                          ? '${_healthData!['gpu_memory_gb'] ?? '?'} GB'
                          : 'Not available',
                    ),
                    _infoRow(
                      'Uptime',
                      '${(_healthData!['uptime_seconds'] as num? ?? 0).round()}s',
                    ),
                    if (_healthData!['adapters_available'] != null)
                      _infoRow(
                        'Adapters',
                        (_healthData!['adapters_available'] as Map).keys.join(', '),
                      ),
                  ],
                ),
              ),
            ),
          ],

          if (_statsData != null) ...[
            const SizedBox(height: 16),
            _sectionHeader('Usage Stats (${appState.tenantId.toUpperCase()})'),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _infoRow(
                      'Total Requests',
                      '${_statsData!['total_requests'] ?? 0}',
                    ),
                    _infoRow(
                      'Avg Latency',
                      '${(_statsData!['avg_latency_ms'] as num? ?? 0).round()}ms',
                    ),
                    _infoRow(
                      'Avg Grounding',
                      '${((_statsData!['avg_grounding_score'] as num? ?? 0) * 100).round()}%',
                    ),
                    _infoRow(
                      'Feedback Rating',
                      '${(_statsData!['avg_feedback_rating'] as num? ?? 0).toStringAsFixed(1)} (${_statsData!['feedback_count'] ?? 0} reviews)',
                    ),
                  ],
                ),
              ),
            ),
          ],

          const SizedBox(height: 32),
          Center(
            child: Text(
              'Multi-Tenant LLM Platform POC v1.0',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade400),
            ),
          ),
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  Widget _sectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8, left: 4),
      child: Text(
        title,
        style: TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.w600,
          color: Colors.grey.shade600,
          letterSpacing: 0.3,
        ),
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(fontSize: 13, color: Colors.grey.shade600)),
          Text(value,
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }
}
