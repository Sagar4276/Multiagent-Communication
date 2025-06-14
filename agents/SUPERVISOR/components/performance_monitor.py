"""
Performance Monitoring Component
Current Date and Time (UTC): 2025-06-14 10:52:09
Current User's Login: Sagar4276

Handles all performance metrics and monitoring.
"""

import time
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from ..core.supervisor_config import SupervisorConfig

@dataclass
class SystemMetrics:
    """System performance metrics"""
    total_messages: int
    successful_responses: int
    failed_responses: int
    rag_requests: int
    chat_requests: int
    system_commands: int
    average_response_time: float
    current_load: float
    memory_usage: Dict[str, Any]
    uptime_seconds: float

class PerformanceMonitor:
    """Handles performance monitoring and metrics"""
    
    def __init__(self, config: SupervisorConfig):
        self.config = config
        self.start_time = time.time()
        
        # Initialize metrics
        self.metrics = SystemMetrics(
            total_messages=0,
            successful_responses=0,
            failed_responses=0,
            rag_requests=0,
            chat_requests=0,
            system_commands=0,
            average_response_time=0.0,
            current_load=0.0,
            memory_usage={},
            uptime_seconds=0.0
        )
        
        # Response time tracking
        self.response_times = []
    
    def update_message_count(self):
        """Update total message count"""
        self.metrics.total_messages += 1
        self.metrics.uptime_seconds = time.time() - self.start_time
    
    def update_rag_request(self):
        """Update RAG request count"""
        self.metrics.rag_requests += 1
    
    def update_chat_request(self):
        """Update chat request count"""
        self.metrics.chat_requests += 1
    
    def update_system_command(self):
        """Update system command count"""
        self.metrics.system_commands += 1
    
    def update_performance_metrics(self, response_time: float, success: bool):
        """Update performance metrics"""
        if success:
            self.metrics.successful_responses += 1
        else:
            self.metrics.failed_responses += 1
        
        # Update response times
        self.response_times.append(response_time)
        if len(self.response_times) > self.config.max_response_history:
            self.response_times = self.response_times[-self.config.max_response_history:]
        
        # Calculate average response time
        if self.response_times:
            self.metrics.average_response_time = sum(self.response_times) / len(self.response_times)
        
        # Update current load estimate
        recent_times = self.response_times[-10:] if len(self.response_times) >= 10 else self.response_times
        if recent_times:
            avg_recent = sum(recent_times) / len(recent_times)
            self.metrics.current_load = min(avg_recent / self.config.max_response_time, 1.0)
    
    def get_success_rate(self) -> float:
        """Calculate current success rate"""
        total = self.metrics.successful_responses + self.metrics.failed_responses
        return self.metrics.successful_responses / total if total > 0 else 1.0
    
    def format_uptime(self, seconds: float) -> str:
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
    
    def format_performance_report(self, user_id: str, current_time: str) -> str:
        """Generate detailed performance report"""
        memory_usage = self._get_memory_usage()
        
        report = f"📊 **ENHANCED SYSTEM PERFORMANCE REPORT**\n\n"
        report += f"👤 **User:** {user_id}\n"
        report += f"🕐 **Report Time:** {current_time} UTC\n"
        report += f"⏱️ **System Uptime:** {self.format_uptime(self.metrics.uptime_seconds)}\n\n"
        
        report += f"📈 **MESSAGE PROCESSING STATISTICS:**\n"
        report += f"   📨 **Total Messages Processed:** {self.metrics.total_messages:,}\n"
        report += f"   ✅ **Successful Responses:** {self.metrics.successful_responses:,}\n"
        report += f"   ❌ **Failed Responses:** {self.metrics.failed_responses:,}\n"
        report += f"   📊 **Success Rate:** {self.get_success_rate():.1%}\n\n"
        
        report += f"🎯 **REQUEST TYPE BREAKDOWN:**\n"
        report += f"   🔍 **RAG/Research Requests:** {self.metrics.rag_requests:,}\n"
        report += f"   💬 **Chat Requests:** {self.metrics.chat_requests:,}\n"
        report += f"   🔧 **System Commands:** {self.metrics.system_commands:,}\n\n"
        
        report += f"⚡ **RESPONSE TIME METRICS:**\n"
        report += f"   📊 **Average Response Time:** {self.metrics.average_response_time:.3f}s\n"
        if self.response_times:
            recent_times = self.response_times[-10:]
            report += f"   🔄 **Recent Average (last 10):** {sum(recent_times)/len(recent_times):.3f}s\n"
            report += f"   ⚡ **Fastest Response:** {min(self.response_times):.3f}s\n"
            report += f"   🐌 **Slowest Response:** {max(self.response_times):.3f}s\n"
        
        report += f"\n🔄 **SYSTEM LOAD:**\n"
        report += f"   📊 **Current Load Estimate:** {self.metrics.current_load:.1%}\n"
        report += f"   🎯 **Load Status:** {'HIGH' if self.metrics.current_load > 0.8 else 'MEDIUM' if self.metrics.current_load > 0.5 else 'LOW'}\n"
        
        report += f"\n💾 **MEMORY PERFORMANCE:**\n"
        report += f"   🔧 **RSS Memory Usage:** {memory_usage.get('rss_mb', 0):.1f}MB\n"
        report += f"   📊 **Memory Percentage:** {memory_usage.get('percentage', 0):.1f}%\n"
        report += f"   💾 **Available Memory:** {memory_usage.get('available_mb', 0):.1f}MB\n"
        
        # Performance grading
        report += f"\n🏆 **PERFORMANCE GRADE:**\n"
        success_rate = self.get_success_rate()
        avg_time = self.metrics.average_response_time
        
        if success_rate >= 0.95 and avg_time <= 2.0:
            grade = "A+ (EXCELLENT)"
        elif success_rate >= 0.90 and avg_time <= 3.0:
            grade = "A (VERY GOOD)"
        elif success_rate >= 0.85 and avg_time <= 5.0:
            grade = "B (GOOD)"
        elif success_rate >= 0.75 and avg_time <= 8.0:
            grade = "C (ACCEPTABLE)"
        else:
            grade = "D (NEEDS IMPROVEMENT)"
        
        report += f"   🎯 **Overall Performance Grade:** {grade}\n"
        
        return report
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percentage': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except ImportError:
            return {
                'rss_mb': 0,
                'vms_mb': 0,
                'percentage': 0,
                'available_mb': 0,
                'note': 'psutil not available'
            }
    
    def get_system_metrics_export(self, current_time: str, version: str, system_name: str, 
                                active_sessions: Dict, health_results: Dict) -> Dict[str, Any]:
        """Export comprehensive system metrics"""
        memory_usage = self._get_memory_usage()
        self.metrics.memory_usage = memory_usage
        
        return {
            'export_info': {
                'timestamp': current_time,
                'supervisor_version': version,
                'system_name': system_name,
                'uptime_seconds': self.metrics.uptime_seconds
            },
            'performance_metrics': asdict(self.metrics),
            'health_diagnostics': health_results,
            'memory_usage': memory_usage,
            'session_data': {
                'active_sessions': len(active_sessions),
                'session_details': {
                    user_id: {
                        'message_count': session['message_count'],
                        'successful_responses': session['successful_responses'],
                        'failed_responses': session['failed_responses'],
                        'session_duration': time.time() - session['start_time'],
                        'last_activity': self._time_since(session['last_activity'])
                    }
                    for user_id, session in active_sessions.items()
                }
            },
            'configuration': asdict(self.config),
            'system_status': {
                'initialized': True,
                'healthy': health_results.get('overall_health', 0) > 0.6,
                'current_health_score': health_results.get('overall_health', 0.0),
                'success_rate': self.get_success_rate()
            }
        }
    
    def _time_since(self, timestamp: float) -> str:
        """Calculate time since timestamp"""
        elapsed = time.time() - timestamp
        return self.format_uptime(elapsed)