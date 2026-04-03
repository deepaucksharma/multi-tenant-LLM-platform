import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'api_service.dart';

class AppState extends ChangeNotifier {
  String _tenantId = 'sis';
  String _serverUrl = 'http://10.0.2.2:8000'; // Android emulator → host
  bool _useRag = true;
  bool _isConnected = false;

  String get tenantId => _tenantId;
  String get serverUrl => _serverUrl;
  bool get useRag => _useRag;
  bool get isConnected => _isConnected;

  String get tenantDisplayName =>
      _tenantId == 'sis' ? 'SIS — Education' : 'MFG — Manufacturing';

  String get tenantEmoji => _tenantId == 'sis' ? '🏫' : '🏭';

  Color get tenantColor =>
      _tenantId == 'sis' ? Colors.blue : Colors.green;

  AppState() {
    _loadPreferences();
  }

  Future<void> _loadPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    _tenantId = prefs.getString('tenant_id') ?? 'sis';
    _serverUrl = prefs.getString('server_url') ?? 'http://10.0.2.2:8000';
    _useRag = prefs.getBool('use_rag') ?? true;
    notifyListeners();
  }

  Future<void> _savePreferences() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('tenant_id', _tenantId);
    await prefs.setString('server_url', _serverUrl);
    await prefs.setBool('use_rag', _useRag);
  }

  void setTenant(String tenant) {
    _tenantId = tenant;
    _savePreferences();
    notifyListeners();
  }

  void setServerUrl(String url) {
    _serverUrl = url.trimRight().replaceAll(RegExp(r'/+$'), '');
    _savePreferences();
    notifyListeners();
  }

  void setUseRag(bool value) {
    _useRag = value;
    _savePreferences();
    notifyListeners();
  }

  Future<void> checkConnection() async {
    try {
      final api = ApiService(baseUrl: _serverUrl);
      final health = await api.healthCheck();
      _isConnected = health['status'] == 'healthy';
    } catch (e) {
      _isConnected = false;
    }
    notifyListeners();
  }
}
