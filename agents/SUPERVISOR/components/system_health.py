"""
System Health Monitoring Component
Current Date and Time (UTC): 2025-06-14 10:52:09
Current User's Login: Sagar4276

Handles all system health diagnostics and monitoring.
"""

import time
from typing import Dict, Any, Optional
from ..core.supervisor_config import SupervisorConfig

class SystemHealthMonitor:
    """Handles system health diagnostics and monitoring"""
    
    def __init__(self, config: SupervisorConfig):
        self.config = config
    
    def run_health_diagnostics(self, shared_memory=None, chat_agent=None) -> Dict[str, Any]:
        """Run comprehensive health diagnostics"""
        diagnostics = {
            'shared_memory': False,
            'chat_agent': False,
            'rag_system': False,
            'memory_usage': 0.0,
            'response_capability': False,
            'overall_health': 0.0
        }
        
        try:
            # Test shared memory
            if shared_memory:
                test_key = f"health_test_{int(time.time())}"
                shared_memory.store_temp_data(test_key, {"health_test": True})
                if shared_memory.get_temp_data(test_key):
                    diagnostics['shared_memory'] = True
                shared_memory.clear_temp_data(test_key)
            
            # Test chat agent
            if chat_agent:
                try:
                    agent_info = chat_agent.get_model_info()
                    if agent_info:
                        diagnostics['chat_agent'] = True
                        diagnostics['rag_system'] = agent_info.get('rag_enabled', False)
                except:
                    diagnostics['chat_agent'] = False
            
            # Test memory usage
            memory_usage = self._get_memory_usage()
            diagnostics['memory_usage'] = memory_usage.get('percentage', 0.0)
            
            # Test response capability
            diagnostics['response_capability'] = diagnostics['shared_memory'] and diagnostics['chat_agent']
            
            # Calculate overall health
            health_score = 0.0
            if diagnostics['shared_memory']:
                health_score += 0.3
            if diagnostics['chat_agent']:
                health_score += 0.4
            if diagnostics['rag_system']:
                health_score += 0.2
            if diagnostics['memory_usage'] < self.config.memory_warning_threshold:
                health_score += 0.1
            
            diagnostics['overall_health'] = health_score
            
        except Exception as e:
            print(f"❌ Health diagnostics failed: {str(e)}")
            diagnostics['overall_health'] = 0.0
        
        return diagnostics
    
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
    
    def get_current_health_status(self, shared_memory=None, chat_agent=None) -> str:
        """Get current system health status"""
        health_results = self.run_health_diagnostics(shared_memory, chat_agent)
        health_score = health_results['overall_health']
        
        if health_score >= 0.8:
            return "EXCELLENT"
        elif health_score >= 0.6:
            return "GOOD"
        elif health_score >= 0.4:
            return "DEGRADED"
        else:
            return "POOR"
    
    def format_health_report(self, user_id: str, current_time: str, shared_memory=None, chat_agent=None) -> str:
        """Generate comprehensive health report"""
        health_results = self.run_health_diagnostics(shared_memory, chat_agent)
        
        report = f"🏥 **ENHANCED SYSTEM HEALTH DIAGNOSTICS REPORT**\n\n"
        report += f"👤 **User:** {user_id}\n"
        report += f"🕐 **Diagnostic Time:** {current_time} UTC\n"
        report += f"🔬 **Health Check Version:** Enhanced v{self.config.version}\n\n"
        
        report += f"📊 **OVERALL HEALTH SCORE:** {health_results['overall_health']:.1%}\n"
        report += f"🎯 **Status:** {self.get_current_health_status(shared_memory, chat_agent)}\n\n"
        
        report += f"🔧 **COMPONENT HEALTH:**\n"
        for component, status in health_results.items():
            if component != 'overall_health':
                if isinstance(status, bool):
                    icon = "✅" if status else "❌"
                    report += f"   {icon} **{component.replace('_', ' ').title()}:** {'HEALTHY' if status else 'ISSUE DETECTED'}\n"
                elif isinstance(status, (int, float)):
                    if component == 'memory_usage':
                        icon = "✅" if status < self.config.memory_warning_threshold else "⚠️" if status < self.config.memory_critical_threshold else "❌"
                        report += f"   {icon} **Memory Usage:** {status:.1f}%\n"
        
        report += f"\n💡 **HEALTH RECOMMENDATIONS:**\n"
        if health_results['overall_health'] >= 0.8:
            report += f"   🟢 System is operating optimally. No action required.\n"
        elif health_results['overall_health'] >= 0.6:
            report += f"   🟡 System is healthy but could benefit from optimization.\n"
            if health_results['memory_usage'] > self.config.memory_warning_threshold:
                report += f"   💡 Consider restarting if memory usage remains high.\n"
        else:
            report += f"   🔴 System health is degraded. Consider the following:\n"
            if not health_results['shared_memory']:
                report += f"   🔧 Restart shared memory system\n"
            if not health_results['chat_agent']:
                report += f"   🔧 Restart chat agent\n"
            if health_results['memory_usage'] > self.config.memory_critical_threshold:
                report += f"   🔧 System restart recommended due to high memory usage\n"
        
        return report