"""
Session Management Component
Current Date and Time (UTC): 2025-06-14 10:52:09
Current User's Login: Sagar4276

Handles user session management and tracking.
"""

import time
from typing import Dict, Any
from ..core.supervisor_config import SupervisorConfig

class SessionManager:
    """Handles user session management and tracking"""
    
    def __init__(self, config: SupervisorConfig):
        self.config = config
        self.active_sessions = {}
        self.session_metrics = {}
    
    def initialize_user_session(self, user_id: str):
        """Initialize or update user session"""
        current_time = time.time()
        
        if user_id not in self.active_sessions:
            self.active_sessions[user_id] = {
                'start_time': current_time,
                'last_activity': current_time,
                'message_count': 0,
                'successful_responses': 0,
                'failed_responses': 0
            }
            self.session_metrics[user_id] = {
                'response_times': [],
                'query_types': [],
                'satisfaction_score': 0.0
            }
        else:
            self.active_sessions[user_id]['last_activity'] = current_time
        
        self.active_sessions[user_id]['message_count'] += 1
    
    def update_session_success(self, user_id: str, success: bool):
        """Update session success metrics"""
        if user_id in self.active_sessions:
            if success:
                self.active_sessions[user_id]['successful_responses'] += 1
            else:
                self.active_sessions[user_id]['failed_responses'] += 1
    
    def add_session_response_time(self, user_id: str, response_time: float):
        """Add response time to session metrics"""
        if user_id in self.session_metrics:
            self.session_metrics[user_id]['response_times'].append(response_time)
            # Keep only last 50 response times per session
            if len(self.session_metrics[user_id]['response_times']) > 50:
                self.session_metrics[user_id]['response_times'] = self.session_metrics[user_id]['response_times'][-50:]
    
    def format_session_report(self, user_id: str, current_time: str) -> str:
        """Generate active sessions monitoring report"""
        report = f"👥 **ACTIVE SESSIONS MONITORING REPORT**\n\n"
        report += f"👤 **Requested by:** {user_id}\n"
        report += f"🕐 **Report Time:** {current_time} UTC\n"
        report += f"🔢 **Total Active Sessions:** {len(self.active_sessions)}\n\n"
        
        if not self.active_sessions:
            report += f"📭 **No active sessions currently.**\n"
            report += f"This could mean:\n"
            report += f"• System was recently restarted\n"
            report += f"• No users have interacted recently\n"
            report += f"• Session timeout has cleared old sessions\n"
            return report
        
        report += f"📊 **SESSION DETAILS:**\n"
        
        for session_user_id, session_data in self.active_sessions.items():
            session_duration = time.time() - session_data['start_time']
            last_activity = time.time() - session_data['last_activity']
            
            report += f"\n👤 **User:** {session_user_id}\n"
            report += f"   🕐 **Session Duration:** {self._format_uptime(session_duration)}\n"
            report += f"   ⏰ **Last Activity:** {self._time_since(session_data['last_activity'])} ago\n"
            report += f"   📨 **Messages:** {session_data['message_count']}\n"
            report += f"   ✅ **Successful:** {session_data['successful_responses']}\n"
            report += f"   ❌ **Failed:** {session_data['failed_responses']}\n"
            
            # Session metrics if available
            if session_user_id in self.session_metrics:
                metrics = self.session_metrics[session_user_id]
                if metrics['response_times']:
                    avg_response = sum(metrics['response_times']) / len(metrics['response_times'])
                    report += f"   ⚡ **Avg Response Time:** {avg_response:.2f}s\n"
        
        # Summary statistics
        total_messages = sum(session['message_count'] for session in self.active_sessions.values())
        total_successful = sum(session['successful_responses'] for session in self.active_sessions.values())
        total_failed = sum(session['failed_responses'] for session in self.active_sessions.values())
        
        report += f"\n📈 **SESSION SUMMARY:**\n"
        report += f"   📨 **Total Messages:** {total_messages:,}\n"
        report += f"   ✅ **Total Successful:** {total_successful:,}\n"
        report += f"   ❌ **Total Failed:** {total_failed:,}\n"
        if total_messages > 0:
            session_success_rate = total_successful / (total_successful + total_failed) if (total_successful + total_failed) > 0 else 1.0
            report += f"   📊 **Session Success Rate:** {session_success_rate:.1%}\n"
            report += f"   📊 **Average Messages per Session:** {total_messages / len(self.active_sessions):.1f}\n"
        
        # Active session health
        active_sessions_count = sum(1 for session in self.active_sessions.values() 
                                  if time.time() - session['last_activity'] < 300)  # Active in last 5 minutes
        
        report += f"\n🔄 **SESSION HEALTH:**\n"
        report += f"   🟢 **Recently Active:** {active_sessions_count}/{len(self.active_sessions)} sessions\n"
        report += f"   📊 **Session Health:** {'EXCELLENT' if active_sessions_count == len(self.active_sessions) else 'GOOD' if active_sessions_count > len(self.active_sessions) * 0.5 else 'DEGRADED'}\n"
        
        return report
    
    def cleanup_inactive_sessions(self) -> int:
        """Clean up inactive sessions and return count of cleaned sessions"""
        current_time = time.time()
        timeout_seconds = self.config.session_timeout_minutes * 60
        inactive_sessions = []
        
        for user_id, session in list(self.active_sessions.items()):
            if current_time - session['last_activity'] > timeout_seconds:
                inactive_sessions.append(user_id)
                del self.active_sessions[user_id]
                if user_id in self.session_metrics:
                    del self.session_metrics[user_id]
        
        return len(inactive_sessions)
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"
    
    def _time_since(self, timestamp: float) -> str:
        """Calculate time since timestamp"""
        elapsed = time.time() - timestamp
        return self._format_uptime(elapsed)