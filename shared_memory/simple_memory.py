# shared_memory/simple_memory.py
import threading
from datetime import datetime
import time

class SimpleSharedMemory:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            'conversations': {},  # user_id -> list of messages
            'agent_flags': {},    # communication flags between agents
            'user_sessions': {}   # active user sessions
        }
    
    def add_message(self, user_id: str, message: str, sender: str):
        """Add a message to conversation history"""
        with self._lock:
            if user_id not in self._data['conversations']:
                self._data['conversations'][user_id] = []
            
            self._data['conversations'][user_id].append({
                'message': message,
                'sender': sender,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    def get_conversation_history(self, user_id: str, limit: int = 5):
        """Get recent conversation history"""
        with self._lock:
            if user_id not in self._data['conversations']:
                return []
            return self._data['conversations'][user_id][-limit:]
    
    def set_flag(self, flag_name: str, value: bool):
        """Set communication flag between agents"""
        with self._lock:
            self._data['agent_flags'][flag_name] = value
    
    def get_flag(self, flag_name: str) -> bool:
        """Get communication flag"""
        with self._lock:
            return self._data['agent_flags'].get(flag_name, False)
    
    def store_temp_data(self, key: str, data):
        """Store temporary data for agent communication"""
        with self._lock:
            self._data[key] = data
    
    def get_temp_data(self, key: str):
        """Get temporary data"""
        with self._lock:
            return self._data.get(key)
    
    def clear_temp_data(self, key: str):
        """Clear temporary data"""
        with self._lock:
            if key in self._data:
                del self._data[key]