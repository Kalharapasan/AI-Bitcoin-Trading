"""
Alert Manager module for Bitcoin trading AI.
Manages real-time alerts, notifications, and monitoring for trading system,
models, risk levels, and market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import json
import pickle
import joblib
from pathlib import Path
import hashlib
import asyncio
from collections import deque, defaultdict
import uuid
import time
import traceback
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl

# Import project modules
from config.settings import AlertSettings, TradingSettings
from config.config_manager import get_config
from core.utils.logger import get_logger
from core.utils.cache import Cache
from core.models.model_manager import ModelManager, ModelMetadata, ModelType, ModelStatus
from core.performance.performance_tracker import PerformanceTracker, PerformanceStatus

# Import notification libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests library not available. Webhook notifications disabled.")

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio not available. SMS notifications disabled.")

try:
    import slack_sdk
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logger.warning("Slack SDK not available. Slack notifications disabled.")

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Enums and Types ============
class AlertSeverity(str, Enum):
    """Severity levels for alerts"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(str, Enum):
    """Types of alerts"""
    MODEL_PERFORMANCE = "model_performance"
    TRADING_SIGNAL = "trading_signal"
    RISK_THRESHOLD = "risk_threshold"
    MARKET_CONDITION = "market_condition"
    SYSTEM_HEALTH = "system_health"
    ACCOUNT_STATUS = "account_status"
    DATA_QUALITY = "data_quality"
    SCHEDULED = "scheduled"
    CUSTOM = "custom"

class AlertStatus(str, Enum):
    """Status of alerts"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    SUPPRESSED = "suppressed"

class NotificationChannel(str, Enum):
    """Notification channels"""
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"
    PUSH = "push"
    DASHBOARD = "dashboard"

# ============ Data Structures ============
@dataclass
class AlertConfig:
    """Configuration for alert management"""
    
    # General settings
    enabled: bool = True
    alert_cooldown_minutes: int = 5  # Minimum time between similar alerts
    max_alerts_per_hour: int = 100
    alert_retention_days: int = 30
    
    # Severity thresholds
    severity_thresholds: Dict[AlertSeverity, Dict[str, float]] = field(default_factory=lambda: {
        AlertSeverity.INFO: {'priority': 1, 'notify': False},
        AlertSeverity.WARNING: {'priority': 2, 'notify': True},
        AlertSeverity.ERROR: {'priority': 3, 'notify': True},
        AlertSeverity.CRITICAL: {'priority': 4, 'notify': True, 'repeat': True}
    })
    
    # Model performance alerts
    model_performance_alerts: Dict[str, float] = field(default_factory=lambda: {
        'accuracy_threshold': 0.6,
        'drift_threshold': 0.3,
        'inference_time_threshold': 1000.0,  # milliseconds
        'prediction_failure_rate': 0.1
    })
    
    # Trading alerts
    trading_alerts: Dict[str, float] = field(default_factory=lambda: {
        'drawdown_threshold': 0.1,  # 10%
        'profit_threshold': 0.05,   # 5%
        'position_size_threshold': 0.2,  # 20% of portfolio
        'volatility_threshold': 0.5,  # 50% annualized
        'stop_loss_trigger': 0.02,  # 2% stop loss
        'take_profit_trigger': 0.05  # 5% take profit
    })
    
    # Risk alerts
    risk_alerts: Dict[str, float] = field(default_factory=lambda: {
        'var_threshold': 0.05,  # 5% VaR
        'cvar_threshold': 0.08,  # 8% CVaR
        'liquidity_threshold': 0.3,
        'concentration_threshold': 0.4
    })
    
    # Market alerts
    market_alerts: Dict[str, float] = field(default_factory=lambda: {
        'price_change_threshold': 0.05,  # 5% price change
        'volume_spike_threshold': 2.0,   # 2x volume
        'volatility_spike_threshold': 2.0,  # 2x volatility
        'correlation_threshold': 0.8
    })
    
    # System health alerts
    system_alerts: Dict[str, float] = field(default_factory=lambda: {
        'memory_threshold': 0.8,  # 80% memory usage
        'cpu_threshold': 0.9,     # 90% CPU usage
        'disk_threshold': 0.9,    # 90% disk usage
        'latency_threshold': 1000.0,  # 1000ms latency
        'error_rate_threshold': 0.01  # 1% error rate
    })
    
    # Notification settings
    notification_channels: List[NotificationChannel] = field(default_factory=lambda: [
        NotificationChannel.LOG,
        NotificationChannel.DASHBOARD
    ])
    
    # Email configuration
    email_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': '',
        'sender_password': '',
        'recipients': []
    })
    
    # SMS configuration (Twilio)
    sms_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'account_sid': '',
        'auth_token': '',
        'from_number': '',
        'recipients': []
    })
    
    # Slack configuration
    slack_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'bot_token': '',
        'channel': '#alerts',
        'username': 'TradingBot'
    })
    
    # Telegram configuration
    telegram_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'bot_token': '',
        'chat_id': '',
        'parse_mode': 'HTML'
    })
    
    # Webhook configuration
    webhook_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'url': '',
        'headers': {'Content-Type': 'application/json'},
        'timeout': 5
    })
    
    # Alert grouping
    group_similar_alerts: bool = True
    group_time_window_minutes: int = 10
    max_group_size: int = 10
    
    # Alert suppression
    enable_suppression: bool = True
    suppression_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Escalation rules
    escalation_rules: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'severity': AlertSeverity.CRITICAL, 'timeout_minutes': 15, 'escalate_to': ['sms', 'phone']},
        {'severity': AlertSeverity.ERROR, 'timeout_minutes': 30, 'escalate_to': ['email']}
    ])
    
    # Scheduling
    scheduled_alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Data storage
    store_alerts: bool = True
    alert_storage_path: str = "data/alerts/"
    backup_frequency_hours: int = 24
    
    def __post_init__(self):
        """Validate configuration"""
        if self.alert_cooldown_minutes < 1:
            raise ValueError("alert_cooldown_minutes must be at least 1")
        
        if self.max_alerts_per_hour < 1:
            raise ValueError("max_alerts_per_hour must be at least 1")
        
        # Create storage directory
        Path(self.alert_storage_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'alert_cooldown_minutes': self.alert_cooldown_minutes,
            'max_alerts_per_hour': self.max_alerts_per_hour,
            'alert_retention_days': self.alert_retention_days,
            'severity_thresholds': {k.value: v for k, v in self.severity_thresholds.items()},
            'model_performance_alerts': self.model_performance_alerts,
            'trading_alerts': self.trading_alerts,
            'risk_alerts': self.risk_alerts,
            'market_alerts': self.market_alerts,
            'system_alerts': self.system_alerts,
            'notification_channels': [c.value for c in self.notification_channels],
            'email_config': self.email_config,
            'sms_config': self.sms_config,
            'slack_config': self.slack_config,
            'telegram_config': self.telegram_config,
            'webhook_config': self.webhook_config,
            'group_similar_alerts': self.group_similar_alerts,
            'group_time_window_minutes': self.group_time_window_minutes,
            'max_group_size': self.max_group_size,
            'enable_suppression': self.enable_suppression,
            'suppression_rules': self.suppression_rules,
            'escalation_rules': self.escalation_rules,
            'scheduled_alerts': self.scheduled_alerts,
            'store_alerts': self.store_alerts,
            'alert_storage_path': self.alert_storage_path,
            'backup_frequency_hours': self.backup_frequency_hours
        }

@dataclass
class Alert:
    """Alert data structure"""
    
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    source: str  # e.g., model_id, trading_system, risk_manager
    
    # Alert content
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Alert data
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Status
    status: AlertStatus = AlertStatus.NEW
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    # Notification
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    notified_at: Optional[datetime] = None
    notification_count: int = 0
    
    # Grouping
    group_id: Optional[str] = None
    is_grouped: bool = False
    group_count: int = 1
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize alert"""
        # Set expiration if not specified
        if self.expires_at is None:
            if self.severity == AlertSeverity.CRITICAL:
                self.expires_at = self.timestamp + timedelta(hours=24)
            elif self.severity == AlertSeverity.ERROR:
                self.expires_at = self.timestamp + timedelta(hours=12)
            elif self.severity == AlertSeverity.WARNING:
                self.expires_at = self.timestamp + timedelta(hours=6)
            else:
                self.expires_at = self.timestamp + timedelta(hours=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'source': self.source,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'data': self.data,
            'metrics': self.metrics,
            'thresholds': self.thresholds,
            'status': self.status.value,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_by': self.resolved_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_notes': self.resolution_notes,
            'notification_channels': [c.value for c in self.notification_channels],
            'notified_at': self.notified_at.isoformat() if self.notified_at else None,
            'notification_count': self.notification_count,
            'group_id': self.group_id,
            'is_grouped': self.is_grouped,
            'group_count': self.group_count,
            'tags': self.tags,
            'created_by': self.created_by,
            'metadata': self.metadata
        }
    
    def acknowledge(self, user: str = "system"):
        """Mark alert as acknowledged"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user
        self.acknowledged_at = datetime.now()
    
    def resolve(self, user: str = "system", notes: str = ""):
        """Mark alert as resolved"""
        self.status = AlertStatus.RESOLVED
        self.resolved_by = user
        self.resolved_at = datetime.now()
        self.resolution_notes = notes
    
    def is_expired(self) -> bool:
        """Check if alert is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def should_notify(self) -> bool:
        """Check if alert should be notified"""
        if self.status in [AlertStatus.RESOLVED, AlertStatus.EXPIRED, AlertStatus.SUPPRESSED]:
            return False
        
        if self.severity == AlertSeverity.INFO:
            return False  # Info alerts typically don't need notification
        
        return True
    
    def get_priority(self) -> int:
        """Get alert priority (higher = more important)"""
        priority_map = {
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4
        }
        return priority_map.get(self.severity, 1)
    
    def format_message(self, format_type: str = "text") -> str:
        """Format alert message for different channels"""
        
        if format_type == "html":
            return f"""
            <h3>{self.title}</h3>
            <p><strong>Severity:</strong> {self.severity.value.upper()}</p>
            <p><strong>Source:</strong> {self.source}</p>
            <p><strong>Time:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>{self.message}</p>
            <p><strong>Status:</strong> {self.status.value}</p>
            {self._format_metrics_html() if self.metrics else ''}
            """
        
        elif format_type == "slack":
            # Slack markdown format
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            emoji = severity_emoji.get(self.severity, "ℹ️")
            
            return f"""
{emoji} *{self.title}* ({self.severity.value.upper()})
*Source:* {self.source}
*Time:* {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
{self.message}
*Status:* {self.status.value}
{self._format_metrics_markdown() if self.metrics else ''}
            """
        
        else:  # Plain text
            return f"""
[{self.severity.value.upper()}] {self.title}
Source: {self.source}
Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
{self.message}
Status: {self.status.value}
{self._format_metrics_text() if self.metrics else ''}
            """
    
    def _format_metrics_html(self) -> str:
        """Format metrics as HTML"""
        if not self.metrics:
            return ""
        
        rows = []
        for key, value in self.metrics.items():
            threshold = self.thresholds.get(key, None)
            if threshold is not None:
                rows.append(f"<tr><td>{key}</td><td>{value:.4f}</td><td>{threshold:.4f}</td></tr>")
            else:
                rows.append(f"<tr><td>{key}</td><td>{value:.4f}</td><td>-</td></tr>")
        
        return f"""
        <table border="1">
            <tr><th>Metric</th><th>Value</th><th>Threshold</th></tr>
            {"".join(rows)}
        </table>
        """
    
    def _format_metrics_markdown(self) -> str:
        """Format metrics as markdown"""
        if not self.metrics:
            return ""
        
        lines = ["*Metrics:*"]
        for key, value in self.metrics.items():
            threshold = self.thresholds.get(key, None)
            if threshold is not None:
                lines.append(f"  - {key}: {value:.4f} (threshold: {threshold:.4f})")
            else:
                lines.append(f"  - {key}: {value:.4f}")
        
        return "\n".join(lines)
    
    def _format_metrics_text(self) -> str:
        """Format metrics as plain text"""
        if not self.metrics:
            return ""
        
        lines = ["Metrics:"]
        for key, value in self.metrics.items():
            threshold = self.thresholds.get(key, None)
            if threshold is not None:
                lines.append(f"  {key}: {value:.4f} (threshold: {threshold:.4f})")
            else:
                lines.append(f"  {key}: {value:.4f}")
        
        return "\n".join(lines)

@dataclass
class AlertGroup:
    """Group of similar alerts"""
    
    group_id: str
    alert_type: AlertType
    severity: AlertSeverity
    source: str
    
    # Group data
    alerts: List[Alert] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_alert_at: datetime = field(default_factory=datetime.now)
    
    # Group statistics
    alert_count: int = 0
    first_alert: Optional[Alert] = None
    last_alert: Optional[Alert] = None
    
    # Status
    status: AlertStatus = AlertStatus.NEW
    is_suppressed: bool = False
    
    def add_alert(self, alert: Alert):
        """Add alert to group"""
        self.alerts.append(alert)
        self.alert_count += 1
        self.last_alert_at = alert.timestamp
        
        if self.first_alert is None:
            self.first_alert = alert
        
        self.last_alert = alert
        
        # Update group status based on worst alert
        if alert.get_priority() > self.get_priority():
            self.severity = alert.severity
        
        # Update group status
        if alert.status == AlertStatus.RESOLVED and all(a.status == AlertStatus.RESOLVED for a in self.alerts):
            self.status = AlertStatus.RESOLVED
        elif alert.status == AlertStatus.ACKNOWLEDGED and all(a.status in [AlertStatus.ACKNOWLEDGED, AlertStatus.RESOLVED] for a in self.alerts):
            self.status = AlertStatus.ACKNOWLEDGED
    
    def get_priority(self) -> int:
        """Get group priority"""
        priority_map = {
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4
        }
        return priority_map.get(self.severity, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'group_id': self.group_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'source': self.source,
            'alert_count': self.alert_count,
            'created_at': self.created_at.isoformat(),
            'last_alert_at': self.last_alert_at.isoformat(),
            'status': self.status.value,
            'is_suppressed': self.is_suppressed,
            'first_alert': self.first_alert.to_dict() if self.first_alert else None,
            'last_alert': self.last_alert.to_dict() if self.last_alert else None,
            'alerts': [alert.to_dict() for alert in self.alerts[-10:]]  # Last 10 alerts
        }

# ============ Notification Handlers ============
class EmailNotifier:
    """Handles email notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{__name__}.EmailNotifier")
        self.smtp_server = None
        
    def send(self, alert: Alert) -> bool:
        """Send email notification"""
        
        if not self.config.get('sender_email') or not self.config.get('sender_password'):
            self.logger.warning("Email configuration incomplete")
            return False
        
        recipients = self.config.get('recipients', [])
        if not recipients:
            self.logger.warning("No email recipients configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.config['sender_email']
            msg['To'] = ', '.join(recipients)
            
            # Create text and HTML versions
            text_part = MIMEText(alert.format_message("text"), 'plain')
            html_part = MIMEText(alert.format_message("html"), 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.config.get('smtp_server', 'smtp.gmail.com'), 
                            self.config.get('smtp_port', 587)) as server:
                server.starttls(context=context)
                server.login(self.config['sender_email'], self.config['sender_password'])
                server.send_message(msg)
            
            self.logger.info(f"Email sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            return False

class SMSNotifier:
    """Handles SMS notifications (using Twilio)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{__name__}.SMSNotifier")
        self.client = None
        
        if TWILIO_AVAILABLE and self.config.get('account_sid') and self.config.get('auth_token'):
            try:
                self.client = TwilioClient(self.config['account_sid'], self.config['auth_token'])
            except Exception as e:
                self.logger.error(f"Failed to initialize Twilio client: {str(e)}")
    
    def send(self, alert: Alert) -> bool:
        """Send SMS notification"""
        
        if self.client is None:
            self.logger.warning("SMS client not initialized")
            return False
        
        recipients = self.config.get('recipients', [])
        if not recipients:
            self.logger.warning("No SMS recipients configured")
            return False
        
        from_number = self.config.get('from_number', '')
        if not from_number:
            self.logger.warning("No sender number configured")
            return False
        
        try:
            message = f"[{alert.severity.value.upper()}] {alert.title}\n{alert.message[:100]}..."
            
            success_count = 0
            for recipient in recipients:
                try:
                    message = self.client.messages.create(
                        body=message,
                        from_=from_number,
                        to=recipient
                    )
                    success_count += 1
                    self.logger.debug(f"SMS sent to {recipient}: {message.sid}")
                except Exception as e:
                    self.logger.error(f"Failed to send SMS to {recipient}: {str(e)}")
            
            self.logger.info(f"SMS sent to {success_count}/{len(recipients)} recipients")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to send SMS: {str(e)}")
            return False

class SlackNotifier:
    """Handles Slack notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{__name__}.SlackNotifier")
        self.client = None
        
        if SLACK_AVAILABLE and self.config.get('bot_token'):
            try:
                self.client = WebClient(token=self.config['bot_token'])
            except Exception as e:
                self.logger.error(f"Failed to initialize Slack client: {str(e)}")
    
    def send(self, alert: Alert) -> bool:
        """Send Slack notification"""
        
        if self.client is None:
            self.logger.warning("Slack client not initialized")
            return False
        
        channel = self.config.get('channel', '#alerts')
        username = self.config.get('username', 'TradingBot')
        
        try:
            # Format message
            message = alert.format_message("slack")
            
            # Send message
            response = self.client.chat_postMessage(
                channel=channel,
                text=message,
                username=username,
                icon_emoji=self._get_severity_emoji(alert.severity)
            )
            
            self.logger.info(f"Slack message sent to {channel}")
            return response['ok']
            
        except SlackApiError as e:
            self.logger.error(f"Slack API error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send Slack message: {str(e)}")
            return False
    
    def _get_severity_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for alert severity"""
        emoji_map = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.ERROR: ":x:",
            AlertSeverity.CRITICAL: ":rotating_light:"
        }
        return emoji_map.get(severity, ":information_source:")

class TelegramNotifier:
    """Handles Telegram notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{__name__}.TelegramNotifier")
        self.base_url = "https://api.telegram.org/bot"
    
    def send(self, alert: Alert) -> bool:
        """Send Telegram notification"""
        
        bot_token = self.config.get('bot_token')
        chat_id = self.config.get('chat_id')
        
        if not bot_token or not chat_id:
            self.logger.warning("Telegram configuration incomplete")
            return False
        
        try:
            # Format message
            message = alert.format_message("text")
            
            # Send message
            url = f"{self.base_url}{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': self.config.get('parse_mode', 'HTML')
            }
            
            if REQUESTS_AVAILABLE:
                response = requests.post(url, json=data, timeout=10)
                
                if response.status_code == 200:
                    self.logger.info(f"Telegram message sent to chat {chat_id}")
                    return True
                else:
                    self.logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                    return False
            else:
                self.logger.warning("Requests library not available")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {str(e)}")
            return False

class WebhookNotifier:
    """Handles webhook notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{__name__}.WebhookNotifier")
    
    def send(self, alert: Alert) -> bool:
        """Send webhook notification"""
        
        url = self.config.get('url')
        if not url:
            self.logger.warning("Webhook URL not configured")
            return False
        
        try:
            headers = self.config.get('headers', {'Content-Type': 'application/json'})
            timeout = self.config.get('timeout', 5)
            
            # Prepare payload
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'system': 'bitcoin_trading_ai'
            }
            
            if REQUESTS_AVAILABLE:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)
                
                if response.status_code in [200, 201, 202, 204]:
                    self.logger.info(f"Webhook sent to {url}")
                    return True
                else:
                    self.logger.error(f"Webhook error: {response.status_code} - {response.text}")
                    return False
            else:
                self.logger.warning("Requests library not available")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook: {str(e)}")
            return False

# ============ Alert Detectors ============
class ModelPerformanceDetector:
    """Detects model performance issues"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{__name__}.ModelDetector")
    
    def detect_issues(self, 
                     model_id: str,
                     metrics: Dict[str, float],
                     metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect model performance issues"""
        
        alerts = []
        
        # Check accuracy threshold
        accuracy = metrics.get('accuracy')
        if accuracy is not None and accuracy < self.config.get('accuracy_threshold', 0.6):
            alerts.append({
                'type': AlertType.MODEL_PERFORMANCE,
                'severity': AlertSeverity.ERROR if accuracy < 0.4 else AlertSeverity.WARNING,
                'title': f"Model {model_id} accuracy below threshold",
                'message': f"Model accuracy ({accuracy:.2%}) is below threshold ({self.config.get('accuracy_threshold', 0.6):.2%})",
                'metrics': {'accuracy': accuracy},
                'thresholds': {'accuracy_threshold': self.config.get('accuracy_threshold', 0.6)},
                'data': {'model_id': model_id, **metadata} if metadata else {'model_id': model_id}
            })
        
        # Check data drift
        data_drift = metrics.get('data_drift_score')
        if data_drift is not None and data_drift > self.config.get('drift_threshold', 0.3):
            alerts.append({
                'type': AlertType.MODEL_PERFORMANCE,
                'severity': AlertSeverity.WARNING,
                'title': f"Model {model_id} data drift detected",
                'message': f"Data drift score ({data_drift:.2%}) exceeds threshold ({self.config.get('drift_threshold', 0.3):.2%})",
                'metrics': {'data_drift_score': data_drift},
                'thresholds': {'drift_threshold': self.config.get('drift_threshold', 0.3)},
                'data': {'model_id': model_id, **metadata} if metadata else {'model_id': model_id}
            })
        
        # Check concept drift
        concept_drift = metrics.get('concept_drift_score')
        if concept_drift is not None and concept_drift > self.config.get('drift_threshold', 0.3):
            alerts.append({
                'type': AlertType.MODEL_PERFORMANCE,
                'severity': AlertSeverity.WARNING,
                'title': f"Model {model_id} concept drift detected",
                'message': f"Concept drift score ({concept_drift:.2%}) exceeds threshold ({self.config.get('drift_threshold', 0.3):.2%})",
                'metrics': {'concept_drift_score': concept_drift},
                'thresholds': {'drift_threshold': self.config.get('drift_threshold', 0.3)},
                'data': {'model_id': model_id, **metadata} if metadata else {'model_id': model_id}
            })
        
        # Check inference time
        inference_time = metrics.get('inference_time_ms')
        if inference_time is not None and inference_time > self.config.get('inference_time_threshold', 1000.0):
            alerts.append({
                'type': AlertType.SYSTEM_HEALTH,
                'severity': AlertSeverity.WARNING,
                'title': f"Model {model_id} slow inference",
                'message': f"Inference time ({inference_time:.1f}ms) exceeds threshold ({self.config.get('inference_time_threshold', 1000.0):.1f}ms)",
                'metrics': {'inference_time_ms': inference_time},
                'thresholds': {'inference_time_threshold': self.config.get('inference_time_threshold', 1000.0)},
                'data': {'model_id': model_id, **metadata} if metadata else {'model_id': model_id}
            })
        
        # Check prediction failure rate
        failure_rate = metrics.get('prediction_failure_rate')
        if failure_rate is not None and failure_rate > self.config.get('prediction_failure_rate', 0.1):
            alerts.append({
                'type': AlertType.MODEL_PERFORMANCE,
                'severity': AlertSeverity.ERROR if failure_rate > 0.2 else AlertSeverity.WARNING,
                'title': f"Model {model_id} high prediction failure rate",
                'message': f"Prediction failure rate ({failure_rate:.2%}) exceeds threshold ({self.config.get('prediction_failure_rate', 0.1):.2%})",
                'metrics': {'prediction_failure_rate': failure_rate},
                'thresholds': {'prediction_failure_rate': self.config.get('prediction_failure_rate', 0.1)},
                'data': {'model_id': model_id, **metadata} if metadata else {'model_id': model_id}
            })
        
        return alerts

class TradingSignalDetector:
    """Detects trading signal alerts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{__name__}.TradingDetector")
    
    def detect_issues(self,
                     signal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect trading signal issues"""
        
        alerts = []
        
        # Check for stop loss trigger
        if 'stop_loss_triggered' in signal_data and signal_data['stop_loss_triggered']:
            alerts.append({
                'type': AlertType.TRADING_SIGNAL,
                'severity': AlertSeverity.WARNING,
                'title': "Stop loss triggered",
                'message': f"Stop loss triggered at price {signal_data.get('trigger_price', 'unknown')}",
                'data': signal_data,
                'tags': ['stop_loss', 'risk_management']
            })
        
        # Check for take profit trigger
        if 'take_profit_triggered' in signal_data and signal_data['take_profit_triggered']:
            alerts.append({
                'type': AlertType.TRADING_SIGNAL,
                'severity': AlertSeverity.INFO,
                'title': "Take profit triggered",
                'message': f"Take profit triggered at price {signal_data.get('trigger_price', 'unknown')}",
                'data': signal_data,
                'tags': ['take_profit', 'profit_booking']
            })
        
        # Check for large position size
        position_size = signal_data.get('position_size')
        if position_size is not None and position_size > self.config.get('position_size_threshold', 0.2):
            alerts.append({
                'type': AlertType.RISK_THRESHOLD,
                'severity': AlertSeverity.WARNING,
                'title': "Large position size detected",
                'message': f"Position size ({position_size:.1%}) exceeds threshold ({self.config.get('position_size_threshold', 0.2):.1%})",
                'metrics': {'position_size': position_size},
                'thresholds': {'position_size_threshold': self.config.get('position_size_threshold', 0.2)},
                'data': signal_data,
                'tags': ['position_size', 'risk_management']
            })
        
        # Check for high volatility
        volatility = signal_data.get('volatility')
        if volatility is not None and volatility > self.config.get('volatility_threshold', 0.5):
            alerts.append({
                'type': AlertType.MARKET_CONDITION,
                'severity': AlertSeverity.WARNING,
                'title': "High market volatility",
                'message': f"Market volatility ({volatility:.1%}) exceeds threshold ({self.config.get('volatility_threshold', 0.5):.1%})",
                'metrics': {'volatility': volatility},
                'thresholds': {'volatility_threshold': self.config.get('volatility_threshold', 0.5)},
                'data': signal_data,
                'tags': ['volatility', 'market_condition']
            })
        
        return alerts

class RiskThresholdDetector:
    """Detects risk threshold breaches"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{__name__}.RiskDetector")
    
    def detect_issues(self,
                     risk_metrics: Dict[str, float],
                     portfolio_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect risk threshold issues"""
        
        alerts = []
        
        # Check VaR threshold
        var_95 = risk_metrics.get('var_95')
        if var_95 is not None and var_95 < -self.config.get('var_threshold', 0.05):
            alerts.append({
                'type': AlertType.RISK_THRESHOLD,
                'severity': AlertSeverity.WARNING if var_95 > -0.1 else AlertSeverity.CRITICAL,
                'title': "Value at Risk (VaR) threshold breached",
                'message': f"95% VaR ({var_95:.2%}) exceeds threshold ({-self.config.get('var_threshold', 0.05):.2%})",
                'metrics': {'var_95': var_95},
                'thresholds': {'var_threshold': self.config.get('var_threshold', 0.05)},
                'data': portfolio_data if portfolio_data else {}
            })
        
        # Check CVaR threshold
        cvar_95 = risk_metrics.get('cvar_95')
        if cvar_95 is not None and cvar_95 < -self.config.get('cvar_threshold', 0.08):
            alerts.append({
                'type': AlertType.RISK_THRESHOLD,
                'severity': AlertSeverity.ERROR if cvar_95 > -0.15 else AlertSeverity.CRITICAL,
                'title': "Conditional VaR (CVaR) threshold breached",
                'message': f"95% CVaR ({cvar_95:.2%}) exceeds threshold ({-self.config.get('cvar_threshold', 0.08):.2%})",
                'metrics': {'cvar_95': cvar_95},
                'thresholds': {'cvar_threshold': self.config.get('cvar_threshold', 0.08)},
                'data': portfolio_data if portfolio_data else {}
            })
        
        # Check drawdown
        drawdown = risk_metrics.get('max_drawdown')
        if drawdown is not None and drawdown < -self.config.get('drawdown_threshold', 0.1):
            alerts.append({
                'type': AlertType.RISK_THRESHOLD,
                'severity': AlertSeverity.WARNING if drawdown > -0.2 else AlertSeverity.CRITICAL,
                'title': "Maximum drawdown threshold breached",
                'message': f"Maximum drawdown ({drawdown:.2%}) exceeds threshold ({-self.config.get('drawdown_threshold', 0.1):.2%})",
                'metrics': {'max_drawdown': drawdown},
                'thresholds': {'drawdown_threshold': self.config.get('drawdown_threshold', 0.1)},
                'data': portfolio_data if portfolio_data else {}
            })
        
        # Check concentration risk
        concentration = risk_metrics.get('concentration_risk')
        if concentration is not None and concentration > self.config.get('concentration_threshold', 0.4):
            alerts.append({
                'type': AlertType.RISK_THRESHOLD,
                'severity': AlertSeverity.WARNING,
                'title': "High portfolio concentration",
                'message': f"Portfolio concentration ({concentration:.2%}) exceeds threshold ({self.config.get('concentration_threshold', 0.4):.2%})",
                'metrics': {'concentration_risk': concentration},
                'thresholds': {'concentration_threshold': self.config.get('concentration_threshold', 0.4)},
                'data': portfolio_data if portfolio_data else {}
            })
        
        return alerts

class MarketConditionDetector:
    """Detects market condition alerts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{__name__}.MarketDetector")
    
    def detect_issues(self,
                     market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect market condition issues"""
        
        alerts = []
        
        # Check for large price change
        price_change = market_data.get('price_change_24h')
        if price_change is not None and abs(price_change) > self.config.get('price_change_threshold', 0.05):
            direction = "increase" if price_change > 0 else "decrease"
            alerts.append({
                'type': AlertType.MARKET_CONDITION,
                'severity': AlertSeverity.WARNING if abs(price_change) < 0.1 else AlertSeverity.ERROR,
                'title': f"Large price {direction} detected",
                'message': f"24h price {direction}: {abs(price_change):.2%}",
                'metrics': {'price_change_24h': price_change},
                'thresholds': {'price_change_threshold': self.config.get('price_change_threshold', 0.05)},
                'data': market_data
            })
        
        # Check for volume spike
        volume_ratio = market_data.get('volume_ratio')
        if volume_ratio is not None and volume_ratio > self.config.get('volume_spike_threshold', 2.0):
            alerts.append({
                'type': AlertType.MARKET_CONDITION,
                'severity': AlertSeverity.WARNING,
                'title': "Volume spike detected",
                'message': f"Volume ratio: {volume_ratio:.1f}x (threshold: {self.config.get('volume_spike_threshold', 2.0):.1f}x)",
                'metrics': {'volume_ratio': volume_ratio},
                'thresholds': {'volume_spike_threshold': self.config.get('volume_spike_threshold', 2.0)},
                'data': market_data
            })
        
        # Check for volatility spike
        volatility_ratio = market_data.get('volatility_ratio')
        if volatility_ratio is not None and volatility_ratio > self.config.get('volatility_spike_threshold', 2.0):
            alerts.append({
                'type': AlertType.MARKET_CONDITION,
                'severity': AlertSeverity.WARNING,
                'title': "Volatility spike detected",
                'message': f"Volatility ratio: {volatility_ratio:.1f}x (threshold: {self.config.get('volatility_spike_threshold', 2.0):.1f}x)",
                'metrics': {'volatility_ratio': volatility_ratio},
                'thresholds': {'volatility_spike_threshold': self.config.get('volatility_spike_threshold', 2.0)},
                'data': market_data
            })
        
        # Check for high correlation
        correlation = market_data.get('correlation')
        if correlation is not None and abs(correlation) > self.config.get('correlation_threshold', 0.8):
            alerts.append({
                'type': AlertType.MARKET_CONDITION,
                'severity': AlertSeverity.INFO,
                'title': "High market correlation detected",
                'message': f"Market correlation: {correlation:.2f} (threshold: {self.config.get('correlation_threshold', 0.8):.2f})",
                'metrics': {'correlation': correlation},
                'thresholds': {'correlation_threshold': self.config.get('correlation_threshold', 0.8)},
                'data': market_data
            })
        
        return alerts

class SystemHealthDetector:
    """Detects system health issues"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{__name__}.SystemDetector")
    
    def detect_issues(self,
                     system_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect system health issues"""
        
        alerts = []
        
        # Check memory usage
        memory_usage = system_metrics.get('memory_usage')
        if memory_usage is not None and memory_usage > self.config.get('memory_threshold', 0.8):
            alerts.append({
                'type': AlertType.SYSTEM_HEALTH,
                'severity': AlertSeverity.WARNING if memory_usage < 0.9 else AlertSeverity.ERROR,
                'title': "High memory usage",
                'message': f"Memory usage: {memory_usage:.1%} (threshold: {self.config.get('memory_threshold', 0.8):.1%})",
                'metrics': {'memory_usage': memory_usage},
                'thresholds': {'memory_threshold': self.config.get('memory_threshold', 0.8)}
            })
        
        # Check CPU usage
        cpu_usage = system_metrics.get('cpu_usage')
        if cpu_usage is not None and cpu_usage > self.config.get('cpu_threshold', 0.9):
            alerts.append({
                'type': AlertType.SYSTEM_HEALTH,
                'severity': AlertSeverity.WARNING,
                'title': "High CPU usage",
                'message': f"CPU usage: {cpu_usage:.1%} (threshold: {self.config.get('cpu_threshold', 0.9):.1%})",
                'metrics': {'cpu_usage': cpu_usage},
                'thresholds': {'cpu_threshold': self.config.get('cpu_threshold', 0.9)}
            })
        
        # Check disk usage
        disk_usage = system_metrics.get('disk_usage')
        if disk_usage is not None and disk_usage > self.config.get('disk_threshold', 0.9):
            alerts.append({
                'type': AlertType.SYSTEM_HEALTH,
                'severity': AlertSeverity.WARNING,
                'title': "High disk usage",
                'message': f"Disk usage: {disk_usage:.1%} (threshold: {self.config.get('disk_threshold', 0.9):.1%})",
                'metrics': {'disk_usage': disk_usage},
                'thresholds': {'disk_threshold': self.config.get('disk_threshold', 0.9)}
            })
        
        # Check latency
        latency = system_metrics.get('latency_ms')
        if latency is not None and latency > self.config.get('latency_threshold', 1000.0):
            alerts.append({
                'type': AlertType.SYSTEM_HEALTH,
                'severity': AlertSeverity.WARNING,
                'title': "High system latency",
                'message': f"System latency: {latency:.1f}ms (threshold: {self.config.get('latency_threshold', 1000.0):.1f}ms)",
                'metrics': {'latency_ms': latency},
                'thresholds': {'latency_threshold': self.config.get('latency_threshold', 1000.0)}
            })
        
        # Check error rate
        error_rate = system_metrics.get('error_rate')
        if error_rate is not None and error_rate > self.config.get('error_rate_threshold', 0.01):
            alerts.append({
                'type': AlertType.SYSTEM_HEALTH,
                'severity': AlertSeverity.ERROR if error_rate > 0.05 else AlertSeverity.WARNING,
                'title': "High error rate",
                'message': f"Error rate: {error_rate:.2%} (threshold: {self.config.get('error_rate_threshold', 0.01):.2%})",
                'metrics': {'error_rate': error_rate},
                'thresholds': {'error_rate_threshold': self.config.get('error_rate_threshold', 0.01)}
            })
        
        return alerts

# ============ Main Alert Manager ============
class AlertManager:
    """Main alert management engine"""
    
    def __init__(self, 
                 config: AlertConfig):
        
        self.config = config
        self.logger = get_logger(__name__)
        
        # Alert storage
        self.alerts: Dict[str, Alert] = {}
        self.alert_groups: Dict[str, AlertGroup] = {}
        
        # Notification handlers
        self.notifiers: Dict[NotificationChannel, Any] = {}
        self._initialize_notifiers()
        
        # Alert detectors
        self.detectors: Dict[AlertType, Any] = {}
        self._initialize_detectors()
        
        # Statistics
        self.alert_count = 0
        self.notification_count = 0
        self.last_alert_time = datetime.now()
        
        # Rate limiting
        self.alert_timestamps: Dict[str, List[datetime]] = defaultdict(list)
        self.grouped_alerts: Dict[str, List[Alert]] = defaultdict(list)
        
        # Alert history
        self.alert_history: deque = deque(maxlen=10000)
        
        # Scheduled alerts
        self.scheduled_alerts: Dict[str, Dict[str, Any]] = {}
        self._load_scheduled_alerts()
        
        # Suppression rules
        self.suppression_rules: List[Dict[str, Any]] = self.config.suppression_rules
        
        self.logger.info("Alert Manager initialized")
    
    def _initialize_notifiers(self):
        """Initialize notification handlers"""
        
        # Email notifier
        if NotificationChannel.EMAIL in self.config.notification_channels:
            if self.config.email_config:
                self.notifiers[NotificationChannel.EMAIL] = EmailNotifier(self.config.email_config)
        
        # SMS notifier
        if NotificationChannel.SMS in self.config.notification_channels:
            if self.config.sms_config:
                self.notifiers[NotificationChannel.SMS] = SMSNotifier(self.config.sms_config)
        
        # Slack notifier
        if NotificationChannel.SLACK in self.config.notification_channels:
            if self.config.slack_config:
                self.notifiers[NotificationChannel.SLACK] = SlackNotifier(self.config.slack_config)
        
        # Telegram notifier
        if NotificationChannel.TELEGRAM in self.config.notification_channels:
            if self.config.telegram_config:
                self.notifiers[NotificationChannel.TELEGRAM] = TelegramNotifier(self.config.telegram_config)
        
        # Webhook notifier
        if NotificationChannel.WEBHOOK in self.config.notification_channels:
            if self.config.webhook_config:
                self.notifiers[NotificationChannel.WEBHOOK] = WebhookNotifier(self.config.webhook_config)
    
    def _initialize_detectors(self):
        """Initialize alert detectors"""
        
        self.detectors[AlertType.MODEL_PERFORMANCE] = ModelPerformanceDetector(
            self.config.model_performance_alerts
        )
        
        self.detectors[AlertType.TRADING_SIGNAL] = TradingSignalDetector(
            self.config.trading_alerts
        )
        
        self.detectors[AlertType.RISK_THRESHOLD] = RiskThresholdDetector(
            self.config.risk_alerts
        )
        
        self.detectors[AlertType.MARKET_CONDITION] = MarketConditionDetector(
            self.config.market_alerts
        )
        
        self.detectors[AlertType.SYSTEM_HEALTH] = SystemHealthDetector(
            self.config.system_alerts
        )
    
    def _load_scheduled_alerts(self):
        """Load scheduled alerts from config"""
        
        for schedule in self.config.scheduled_alerts:
            schedule_id = schedule.get('id', str(uuid.uuid4()))
            self.scheduled_alerts[schedule_id] = schedule
    
    def create_alert(self,
                    alert_type: AlertType,
                    severity: AlertSeverity,
                    source: str,
                    title: str,
                    message: str,
                    data: Optional[Dict[str, Any]] = None,
                    metrics: Optional[Dict[str, float]] = None,
                    thresholds: Optional[Dict[str, float]] = None,
                    tags: Optional[List[str]] = None) -> Alert:
        """Create a new alert"""
        
        # Generate alert ID
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            source=source,
            title=title,
            message=message,
            data=data or {},
            metrics=metrics or {},
            thresholds=thresholds or {},
            tags=tags or [],
            notification_channels=self.config.notification_channels.copy()
        )
        
        return alert
    
    def process_alert(self, alert: Alert) -> bool:
        """Process an alert (check suppression, grouping, rate limiting)"""
        
        if not self.config.enabled:
            self.logger.debug("Alert processing disabled")
            return False
        
        # Check if alert should be suppressed
        if self.config.enable_suppression and self._should_suppress_alert(alert):
            alert.status = AlertStatus.SUPPRESSED
            self.logger.debug(f"Alert suppressed: {alert.alert_id}")
            return False
        
        # Check rate limiting
        if not self._check_rate_limit(alert):
            self.logger.debug(f"Alert rate limited: {alert.alert_id}")
            return False
        
        # Check for similar alerts to group
        if self.config.group_similar_alerts:
            group_id = self._get_group_id(alert)
            if group_id in self.grouped_alerts:
                # Check if we should add to existing group
                recent_alerts = self.grouped_alerts[group_id]
                time_window = timedelta(minutes=self.config.group_time_window_minutes)
                
                recent_relevant = [a for a in recent_alerts 
                                  if (alert.timestamp - a.timestamp) <= time_window]
                
                if len(recent_relevant) > 0:
                    # Group with existing alerts
                    return self._add_to_group(alert, group_id, recent_relevant)
        
        # Process as standalone alert
        return self._process_standalone_alert(alert)
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        
        for rule in self.suppression_rules:
            # Check if rule matches alert
            matches = True
            
            # Check type
            if 'type' in rule and rule['type'] != alert.alert_type.value:
                matches = False
            
            # Check severity
            if 'severity' in rule and rule['severity'] != alert.severity.value:
                matches = False
            
            # Check source pattern
            if 'source_pattern' in rule:
                import re
                if not re.search(rule['source_pattern'], alert.source):
                    matches = False
            
            # Check time-based suppression
            if matches and 'suppress_until' in rule:
                suppress_until = datetime.fromisoformat(rule['suppress_until'])
                if datetime.now() < suppress_until:
                    return True
        
        return False
    
    def _check_rate_limit(self, alert: Alert) -> bool:
        """Check rate limiting for alerts"""
        
        # Check overall alert rate
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        hour_alerts = [t for t in self.alert_timestamps['_all'] 
                      if t >= current_hour]
        
        if len(hour_alerts) >= self.config.max_alerts_per_hour:
            self.logger.warning(f"Maximum alerts per hour reached: {self.config.max_alerts_per_hour}")
            return False
        
        # Check cooldown for similar alerts
        alert_key = f"{alert.alert_type.value}_{alert.source}"
        cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
        
        recent_alerts = [t for t in self.alert_timestamps[alert_key]
                        if (datetime.now() - t) <= cooldown]
        
        if recent_alerts:
            self.logger.debug(f"Alert cooldown active for {alert_key}")
            return False
        
        # Update timestamps
        now = datetime.now()
        self.alert_timestamps['_all'].append(now)
        self.alert_timestamps[alert_key].append(now)
        
        # Clean old timestamps
        for key in list(self.alert_timestamps.keys()):
            # Keep only timestamps from last 24 hours
            cutoff = datetime.now() - timedelta(hours=24)
            self.alert_timestamps[key] = [t for t in self.alert_timestamps[key] 
                                         if t >= cutoff]
        
        return True
    
    def _get_group_id(self, alert: Alert) -> str:
        """Generate group ID for alert"""
        
        # Create hash based on alert characteristics
        group_key = f"{alert.alert_type.value}_{alert.source}_{alert.severity.value}"
        
        # Include important metrics for grouping
        if alert.metrics:
            important_metrics = ['accuracy', 'data_drift_score', 'max_drawdown', 'var_95']
            for metric in important_metrics:
                if metric in alert.metrics:
                    group_key += f"_{metric}_{alert.metrics[metric]:.3f}"
        
        return hashlib.md5(group_key.encode()).hexdigest()[:16]
    
    def _add_to_group(self, 
                     alert: Alert, 
                     group_id: str, 
                     recent_alerts: List[Alert]) -> bool:
        """Add alert to existing group"""
        
        if group_id not in self.alert_groups:
            # Create new group
            group = AlertGroup(
                group_id=group_id,
                alert_type=alert.alert_type,
                severity=alert.severity,
                source=alert.source
            )
            self.alert_groups[group_id] = group
        
        # Add alert to group
        alert.group_id = group_id
        alert.is_grouped = True
        
        self.alert_groups[group_id].add_alert(alert)
        
        # Update group count on alert
        alert.group_count = self.alert_groups[group_id].alert_count
        
        # Store alert
        self.alerts[alert.alert_id] = alert
        self.grouped_alerts[group_id].append(alert)
        
        # Send notification if this is the first alert in group or severity increased
        if self.alert_groups[group_id].alert_count == 1:
            return self._send_notification(alert)
        elif alert.severity.value != self.alert_groups[group_id].severity.value:
            # Severity changed, send notification
            return self._send_notification(alert)
        
        self.logger.debug(f"Alert added to group {group_id}, total: {self.alert_groups[group_id].alert_count}")
        return True
    
    def _process_standalone_alert(self, alert: Alert) -> bool:
        """Process standalone alert"""
        
        # Store alert
        self.alerts[alert.alert_id] = alert
        self.alert_count += 1
        self.last_alert_time = alert.timestamp
        
        # Add to history
        self.alert_history.append(alert)
        
        # Initialize group for potential future grouping
        group_id = self._get_group_id(alert)
        self.grouped_alerts[group_id] = [alert]
        
        # Send notifications
        success = self._send_notification(alert)
        
        if success:
            alert.notified_at = datetime.now()
            alert.notification_count += 1
        
        # Store alert if configured
        if self.config.store_alerts:
            self._store_alert(alert)
        
        self.logger.info(f"Alert created: {alert.alert_id} [{alert.severity.value}] {alert.title}")
        
        return success
    
    def _send_notification(self, alert: Alert) -> bool:
        """Send notifications through configured channels"""
        
        if not alert.should_notify():
            return False
        
        success = False
        notification_channels = alert.notification_channels or self.config.notification_channels
        
        for channel in notification_channels:
            if channel in self.notifiers:
                try:
                    channel_success = self.notifiers[channel].send(alert)
                    if channel_success:
                        success = True
                        self.notification_count += 1
                        self.logger.debug(f"Notification sent via {channel.value}")
                except Exception as e:
                    self.logger.error(f"Failed to send notification via {channel.value}: {str(e)}")
        
        # Always log alerts
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ALERT: {alert.title} - {alert.message}")
        elif alert.severity == AlertSeverity.ERROR:
            self.logger.error(f"ERROR ALERT: {alert.title} - {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            self.logger.warning(f"WARNING ALERT: {alert.title} - {alert.message}")
        else:
            self.logger.info(f"INFO ALERT: {alert.title} - {alert.message}")
        
        return success
    
    def _store_alert(self, alert: Alert):
        """Store alert to persistent storage"""
        
        try:
            alert_dir = Path(self.config.alert_storage_path)
            alert_dir.mkdir(parents=True, exist_ok=True)
            
            # Store by date
            date_str = alert.timestamp.strftime('%Y-%m-%d')
            date_file = alert_dir / f"alerts_{date_str}.json"
            
            alerts_data = []
            if date_file.exists():
                with open(date_file, 'r') as f:
                    alerts_data = json.load(f)
            
            alerts_data.append(alert.to_dict())
            
            # Keep only recent alerts in file
            max_alerts_per_day = 1000
            if len(alerts_data) > max_alerts_per_day:
                alerts_data = alerts_data[-max_alerts_per_day:]
            
            with open(date_file, 'w') as f:
                json.dump(alerts_data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to store alert: {str(e)}")
    
    def detect_model_performance_issues(self,
                                       model_id: str,
                                       metrics: Dict[str, float],
                                       metadata: Optional[Dict[str, Any]] = None) -> List[Alert]:
        """Detect and create model performance alerts"""
        
        if AlertType.MODEL_PERFORMANCE not in self.detectors:
            return []
        
        detector = self.detectors[AlertType.MODEL_PERFORMANCE]
        alert_data_list = detector.detect_issues(model_id, metrics, metadata)
        
        alerts = []
        for alert_data in alert_data_list:
            alert = self.create_alert(
                alert_type=alert_data['type'],
                severity=alert_data['severity'],
                source=f"model:{model_id}",
                title=alert_data['title'],
                message=alert_data['message'],
                data=alert_data.get('data', {}),
                metrics=alert_data.get('metrics', {}),
                thresholds=alert_data.get('thresholds', {}),
                tags=alert_data.get('tags', [])
            )
            
            if self.process_alert(alert):
                alerts.append(alert)
        
        return alerts
    
    def detect_trading_signal_issues(self,
                                    signal_data: Dict[str, Any]) -> List[Alert]:
        """Detect and create trading signal alerts"""
        
        if AlertType.TRADING_SIGNAL not in self.detectors:
            return []
        
        detector = self.detectors[AlertType.TRADING_SIGNAL]
        alert_data_list = detector.detect_issues(signal_data)
        
        alerts = []
        for alert_data in alert_data_list:
            alert = self.create_alert(
                alert_type=alert_data['type'],
                severity=alert_data['severity'],
                source="trading_system",
                title=alert_data['title'],
                message=alert_data['message'],
                data=alert_data.get('data', {}),
                metrics=alert_data.get('metrics', {}),
                thresholds=alert_data.get('thresholds', {}),
                tags=alert_data.get('tags', [])
            )
            
            if self.process_alert(alert):
                alerts.append(alert)
        
        return alerts
    
    def detect_risk_threshold_issues(self,
                                    risk_metrics: Dict[str, float],
                                    portfolio_data: Optional[Dict[str, Any]] = None) -> List[Alert]:
        """Detect and create risk threshold alerts"""
        
        if AlertType.RISK_THRESHOLD not in self.detectors:
            return []
        
        detector = self.detectors[AlertType.RISK_THRESHOLD]
        alert_data_list = detector.detect_issues(risk_metrics, portfolio_data)
        
        alerts = []
        for alert_data in alert_data_list:
            alert = self.create_alert(
                alert_type=alert_data['type'],
                severity=alert_data['severity'],
                source="risk_manager",
                title=alert_data['title'],
                message=alert_data['message'],
                data=alert_data.get('data', {}),
                metrics=alert_data.get('metrics', {}),
                thresholds=alert_data.get('thresholds', {}),
                tags=alert_data.get('tags', [])
            )
            
            if self.process_alert(alert):
                alerts.append(alert)
        
        return alerts
    
    def detect_market_condition_issues(self,
                                      market_data: Dict[str, Any]) -> List[Alert]:
        """Detect and create market condition alerts"""
        
        if AlertType.MARKET_CONDITION not in self.detectors:
            return []
        
        detector = self.detectors[AlertType.MARKET_CONDITION]
        alert_data_list = detector.detect_issues(market_data)
        
        alerts = []
        for alert_data in alert_data_list:
            alert = self.create_alert(
                alert_type=alert_data['type'],
                severity=alert_data['severity'],
                source="market_data",
                title=alert_data['title'],
                message=alert_data['message'],
                data=alert_data.get('data', {}),
                metrics=alert_data.get('metrics', {}),
                thresholds=alert_data.get('thresholds', {}),
                tags=alert_data.get('tags', [])
            )
            
            if self.process_alert(alert):
                alerts.append(alert)
        
        return alerts
    
    def detect_system_health_issues(self,
                                   system_metrics: Dict[str, float]) -> List[Alert]:
        """Detect and create system health alerts"""
        
        if AlertType.SYSTEM_HEALTH not in self.detectors:
            return []
        
        detector = self.detectors[AlertType.SYSTEM_HEALTH]
        alert_data_list = detector.detect_issues(system_metrics)
        
        alerts = []
        for alert_data in alert_data_list:
            alert = self.create_alert(
                alert_type=alert_data['type'],
                severity=alert_data['severity'],
                source="system_monitor",
                title=alert_data['title'],
                message=alert_data['message'],
                data=alert_data.get('data', {}),
                metrics=alert_data.get('metrics', {}),
                thresholds=alert_data.get('thresholds', {}),
                tags=alert_data.get('tags', [])
            )
            
            if self.process_alert(alert):
                alerts.append(alert)
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        
        if alert_id not in self.alerts:
            self.logger.warning(f"Alert not found: {alert_id}")
            return False
        
        alert = self.alerts[alert_id]
        alert.acknowledge(user)
        
        # If alert is in a group, acknowledge the group
        if alert.group_id and alert.group_id in self.alert_groups:
            group = self.alert_groups[alert.group_id]
            group.status = AlertStatus.ACKNOWLEDGED
        
        self.logger.info(f"Alert acknowledged: {alert_id} by {user}")
        return True
    
    def resolve_alert(self, 
                     alert_id: str, 
                     user: str = "system", 
                     notes: str = "") -> bool:
        """Resolve an alert"""
        
        if alert_id not in self.alerts:
            self.logger.warning(f"Alert not found: {alert_id}")
            return False
        
        alert = self.alerts[alert_id]
        alert.resolve(user, notes)
        
        # If alert is in a group, check if all alerts in group are resolved
        if alert.group_id and alert.group_id in self.alert_groups:
            group = self.alert_groups[alert.group_id]
            if all(a.status == AlertStatus.RESOLVED for a in group.alerts):
                group.status = AlertStatus.RESOLVED
        
        self.logger.info(f"Alert resolved: {alert_id} by {user}")
        return True
    
    def get_alerts(self,
                  status: Optional[AlertStatus] = None,
                  severity: Optional[AlertSeverity] = None,
                  alert_type: Optional[AlertType] = None,
                  source: Optional[str] = None,
                  limit: int = 100) -> List[Alert]:
        """Get alerts with optional filtering"""
        
        alerts = list(self.alerts.values())
        
        # Apply filters
        if status is not None:
            alerts = [a for a in alerts if a.status == status]
        
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type is not None:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if source is not None:
            alerts = [a for a in alerts if source in a.source]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts[:limit]
    
    def get_alert_groups(self,
                        status: Optional[AlertStatus] = None,
                        severity: Optional[AlertSeverity] = None,
                        limit: int = 50) -> List[AlertGroup]:
        """Get alert groups with optional filtering"""
        
        groups = list(self.alert_groups.values())
        
        # Apply filters
        if status is not None:
            groups = [g for g in groups if g.status == status]
        
        if severity is not None:
            groups = [g for g in groups if g.severity == severity]
        
        # Sort by last alert time (newest first)
        groups.sort(key=lambda x: x.last_alert_at, reverse=True)
        
        return groups[:limit]
    
    def get_active_alerts_count(self) -> Dict[str, int]:
        """Get count of active alerts by severity"""
        
        counts = {
            'total': 0,
            'critical': 0,
            'error': 0,
            'warning': 0,
            'info': 0
        }
        
        for alert in self.alerts.values():
            if alert.status not in [AlertStatus.RESOLVED, AlertStatus.EXPIRED]:
                counts['total'] += 1
                
                if alert.severity == AlertSeverity.CRITICAL:
                    counts['critical'] += 1
                elif alert.severity == AlertSeverity.ERROR:
                    counts['error'] += 1
                elif alert.severity == AlertSeverity.WARNING:
                    counts['warning'] += 1
                elif alert.severity == AlertSeverity.INFO:
                    counts['info'] += 1
        
        return counts
    
    def cleanup_expired_alerts(self):
        """Clean up expired alerts"""
        
        expired_count = 0
        current_time = datetime.now()
        
        for alert_id, alert in list(self.alerts.items()):
            if alert.is_expired():
                alert.status = AlertStatus.EXPIRED
                expired_count += 1
        
        # Clean up old grouped alerts
        cutoff_time = current_time - timedelta(minutes=self.config.group_time_window_minutes * 2)
        for group_id, alerts in list(self.grouped_alerts.items()):
            self.grouped_alerts[group_id] = [
                a for a in alerts if a.timestamp >= cutoff_time
            ]
            if not self.grouped_alerts[group_id]:
                del self.grouped_alerts[group_id]
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired alerts")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert manager statistics"""
        
        active_counts = self.get_active_alerts_count()
        
        return {
            'total_alerts_created': self.alert_count,
            'active_alerts': active_counts,
            'notification_count': self.notification_count,
            'last_alert_time': self.last_alert_time.isoformat(),
            'alert_groups_count': len(self.alert_groups),
            'notifiers_configured': list(self.notifiers.keys()),
            'detectors_configured': list(self.detectors.keys()),
            'rate_limited': len([t for t in self.alert_timestamps['_all'] 
                               if t >= datetime.now() - timedelta(hours=1)])
        }
    
    def run_scheduled_alerts(self):
        """Run scheduled alert checks"""
        
        current_time = datetime.now()
        
        for schedule_id, schedule in self.scheduled_alerts.items():
            # Check if it's time to run this schedule
            schedule_type = schedule.get('type', 'daily')
            last_run = schedule.get('last_run')
            
            should_run = False
            
            if schedule_type == 'daily':
                # Run once per day
                if not last_run or (current_time - datetime.fromisoformat(last_run)).days >= 1:
                    should_run = True
            
            elif schedule_type == 'hourly':
                # Run once per hour
                if not last_run or (current_time - datetime.fromisoformat(last_run)).hours >= 1:
                    should_run = True
            
            elif schedule_type == 'weekly':
                # Run once per week
                if not last_run or (current_time - datetime.fromisoformat(last_run)).days >= 7:
                    should_run = True
            
            if should_run:
                # Create scheduled alert
                alert = self.create_alert(
                    alert_type=AlertType.SCHEDULED,
                    severity=AlertSeverity.INFO,
                    source="scheduler",
                    title=schedule.get('title', 'Scheduled Alert'),
                    message=schedule.get('message', 'Scheduled check completed'),
                    data={'schedule_id': schedule_id}
                )
                
                self.process_alert(alert)
                
                # Update last run time
                schedule['last_run'] = current_time.isoformat()
    
    def run_periodic_tasks(self):
        """Run periodic maintenance tasks"""
        
        # Clean up expired alerts
        self.cleanup_expired_alerts()
        
        # Run scheduled alerts
        self.run_scheduled_alerts()
        
        # Check for escalation
        self._check_escalations()
    
    def _check_escalations(self):
        """Check for alerts that need escalation"""
        
        for rule in self.config.escalation_rules:
            severity = AlertSeverity(rule['severity'])
            timeout = timedelta(minutes=rule['timeout_minutes'])
            escalate_to = rule.get('escalate_to', [])
            
            # Find unacknowledged alerts of this severity
            unacknowledged = [
                a for a in self.alerts.values()
                if a.severity == severity 
                and a.status == AlertStatus.NEW
                and (datetime.now() - a.timestamp) > timeout
            ]
            
            for alert in unacknowledged:
                # Add additional notification channels
                for channel_name in escalate_to:
                    try:
                        channel = NotificationChannel(channel_name)
                        if channel not in alert.notification_channels:
                            alert.notification_channels.append(channel)
                    except ValueError:
                        self.logger.warning(f"Unknown notification channel: {channel_name}")
                
                # Resend notification with additional channels
                self._send_notification(alert)
                
                self.logger.info(f"Alert {alert.alert_id} escalated to {escalate_to}")

# ============ Helper Functions ============
def create_alert_manager(config: Optional[AlertConfig] = None) -> AlertManager:
    """Factory function to create alert manager"""
    
    if config is None:
        config = AlertConfig()
    
    return AlertManager(config)


def load_alert_manager_from_config(config_path: str) -> AlertManager:
    """Load alert manager from configuration file"""
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert string enums back to Enum types
    if 'notification_channels' in config_dict:
        config_dict['notification_channels'] = [
            NotificationChannel(c) for c in config_dict['notification_channels']
        ]
    
    if 'severity_thresholds' in config_dict:
        config_dict['severity_thresholds'] = {
            AlertSeverity(k): v for k, v in config_dict['severity_thresholds'].items()
        }
    
    config = AlertConfig(**config_dict)
    return AlertManager(config)


def create_test_alert(alert_manager: AlertManager,
                     severity: AlertSeverity = AlertSeverity.WARNING) -> Alert:
    """Create a test alert"""
    
    alert = alert_manager.create_alert(
        alert_type=AlertType.SYSTEM_HEALTH,
        severity=severity,
        source="test_system",
        title="Test Alert",
        message="This is a test alert to verify the alert system is working.",
        data={'test': True, 'timestamp': datetime.now().isoformat()},
        tags=['test', 'system']
    )
    
    alert_manager.process_alert(alert)
    return alert


def export_alerts_to_dataframe(alerts: List[Alert]) -> pd.DataFrame:
    """Export alerts to pandas DataFrame"""
    
    if not alerts:
        return pd.DataFrame()
    
    data = []
    for alert in alerts:
        alert_dict = alert.to_dict()
        data.append(alert_dict)
    
    df = pd.DataFrame(data)
    
    # Convert timestamp strings to datetime
    timestamp_cols = ['timestamp', 'expires_at', 'acknowledged_at', 'resolved_at', 'notified_at']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


def visualize_alert_trends(alerts_df: pd.DataFrame, 
                          time_column: str = 'timestamp',
                          severity_column: str = 'severity') -> Optional[Any]:
    """Visualize alert trends over time"""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if alerts_df.empty:
            return None
        
        # Set style
        plt.style.use('seaborn')
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Alerts over time
        alerts_by_date = alerts_df.groupby(alerts_df[time_column].dt.date).size()
        axes[0, 0].plot(alerts_by_date.index, alerts_by_date.values, marker='o')
        axes[0, 0].set_title('Alerts Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Alerts')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Severity distribution
        severity_counts = alerts_df[severity_column].value_counts()
        axes[0, 1].bar(severity_counts.index, severity_counts.values)
        axes[0, 1].set_title('Alert Severity Distribution')
        axes[0, 1].set_xlabel('Severity')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Alert type distribution
        if 'alert_type' in alerts_df.columns:
            type_counts = alerts_df['alert_type'].value_counts().head(10)
            axes[1, 0].barh(type_counts.index, type_counts.values)
            axes[1, 0].set_title('Top 10 Alert Types')
            axes[1, 0].set_xlabel('Count')
        
        # 4. Status distribution
        if 'status' in alerts_df.columns:
            status_counts = alerts_df['status'].value_counts()
            axes[1, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Alert Status Distribution')
        
        plt.tight_layout()
        return fig
        
    except ImportError:
        logger.warning("Matplotlib not available for visualization")
        return None
    except Exception as e:
        logger.error(f"Failed to visualize alert trends: {str(e)}")
        return None


# ============ Example Usage ============
if __name__ == "__main__":
    # Example usage
    print("Alert Manager Module")
    
    # Create a sample config
    config = AlertConfig(
        enabled=True,
        notification_channels=[NotificationChannel.LOG, NotificationChannel.DASHBOARD],
        model_performance_alerts={
            'accuracy_threshold': 0.6,
            'drift_threshold': 0.3
        }
    )
    
    # Create alert manager
    alert_manager = AlertManager(config)
    
    print(f"Alert Manager initialized")
    print(f"Notification channels: {[c.value for c in config.notification_channels]}")
    
    # Create a test alert
    test_alert = create_test_alert(alert_manager)
    
    print(f"Test alert created: {test_alert.alert_id}")
    print(f"Alert title: {test_alert.title}")
    print(f"Alert severity: {test_alert.severity.value}")
    
    # Get statistics
    stats = alert_manager.get_statistics()
    print(f"Total alerts: {stats['total_alerts_created']}")
    print(f"Active alerts: {stats['active_alerts']}")