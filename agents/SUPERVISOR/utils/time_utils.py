"""
Time Utilities
Current Date and Time (UTC): 2025-06-14 10:52:09
Current User's Login: Sagar4276

Handles time formatting and utilities.
"""

from datetime import datetime, timezone

class TimeUtils:
    """Time formatting and utilities"""
    
    @staticmethod
    def get_current_time() -> str:
        """Get current UTC time with consistent formatting"""
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def format_uptime(seconds: float) -> str:
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
    
    @staticmethod
    def time_since(timestamp: float) -> str:
        """Calculate time since timestamp"""
        import time
        elapsed = time.time() - timestamp
        return TimeUtils.format_uptime(elapsed)