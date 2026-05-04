from dash import html
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

def create_sidebar():
    """
    Creates the main sidebar navigation for the dashboard.
    """
    return html.Div(
        className="sidebar",
        children=[
            # User info section
            create_user_info_section(),
            
            # Main navigation
            create_navigation_section(),
            
            # Quick actions
            create_quick_actions_section(),
            
            # Model status
            create_model_status_section(),
            
            # System info
            create_system_info_section()
        ]
    )


def create_user_info_section():
    """
    Creates the user information section in the sidebar.
    """
    return html.Div(
        className="sidebar-user-section",
        children=[
            html.Div(
                className="user-avatar-container",
                children=[
                    html.Img(
                        src="/assets/user-avatar.png",
                        alt="User Avatar",
                        className="user-avatar-large"
                    ),
                    html.Div(
                        className="user-status-indicator",
                        id="user-status-indicator"
                    )
                ]
            ),
            html.H3("Administrator", className="user-name"),
            html.P("System Admin", className="user-role"),
            html.Div(
                className="user-balance",
                children=[
                    html.P("Account Balance", className="balance-label"),
                    html.H4(
                        id="user-balance-amount",
                        className="balance-amount",
                        children="$0.00"
                    ),
                    html.P(
                        id="balance-change",
                        className="balance-change positive",
                        children="+0.00%"
                    )
                ]
            )
        ]
    )


def create_navigation_section():
    """
    Creates the main navigation menu in the sidebar.
    """
    return html.Nav(
        className="sidebar-nav",
        children=[
            html.Ul(
                className="nav-menu",
                children=[
                    create_nav_item(
                        icon="material-symbols:dashboard",
                        label="Dashboard",
                        value="dashboard",
                        is_active=True
                    ),
                    create_nav_item(
                        icon="material-symbols:trending-up",
                        label="Trading",
                        value="trading"
                    ),
                    create_nav_item(
                        icon="material-symbols:model-training",
                        label="Models",
                        value="models"
                    ),
                    create_nav_item(
                        icon="material-symbols:analytics",
                        label="Backtesting",
                        value="backtesting"
                    ),
                    create_nav_item(
                        icon="material-symbols:monitoring",
                        label="Monitoring",
                        value="monitoring"
                    ),
                    create_nav_item(
                        icon="material-symbols:show-chart",
                        label="Charts",
                        value="charts"
                    ),
                    create_nav_item(
                        icon="material-symbols:settings",
                        label="Settings",
                        value="settings"
                    )
                ]
            )
        ]
    )


def create_nav_item(icon, label, value, is_active=False):
    """
    Creates a single navigation item.
    
    Args:
        icon: Icon name from iconify
        label: Display label
        value: Value for identification
        is_active: Whether the item is active
    
    Returns:
        html.Li: Navigation item
    """
    return html.Li(
        className=f"nav-item {'active' if is_active else ''}",
        children=[
            html.A(
                href=f"#{value}",
                className="nav-link",
                **{"data-value": value},
                children=[
                    DashIconify(
                        icon=icon,
                        className="nav-icon",
                        width=20,
                        height=20
                    ),
                    html.Span(label, className="nav-label"),
                    html.Span(
                        className="notification-badge",
                        id=f"nav-badge-{value}",
                        style={"display": "none"}
                    )
                ]
            )
        ]
    )


def create_quick_actions_section():
    """
    Creates the quick actions section in the sidebar.
    """
    return html.Div(
        className="sidebar-actions",
        children=[
            html.H5("Quick Actions", className="actions-title"),
            dbc.ButtonGroup(
                className="action-buttons",
                children=[
                    dbc.Button(
                        [
                            DashIconify(icon="material-symbols:play-arrow", width=16),
                            " Start Trading"
                        ],
                        id="start-trading-btn",
                        color="success",
                        className="action-btn",
                        size="sm"
                    ),
                    dbc.Button(
                        [
                            DashIconify(icon="material-symbols:stop", width=16),
                            " Stop Trading"
                        ],
                        id="stop-trading-btn",
                        color="danger",
                        className="action-btn",
                        size="sm"
                    )
                ],
                vertical=True
            ),
            dbc.ButtonGroup(
                className="action-buttons",
                children=[
                    dbc.Button(
                        [
                            DashIconify(icon="material-symbols:refresh", width=16),
                            " Retrain Model"
                        ],
                        id="retrain-model-btn",
                        color="primary",
                        className="action-btn",
                        size="sm"
                    ),
                    dbc.Button(
                        [
                            DashIconify(icon="material-symbols:download", width=16),
                            " Export Data"
                        ],
                        id="export-data-btn",
                        color="info",
                        className="action-btn",
                        size="sm"
                    )
                ],
                vertical=True
            )
        ]
    )


def create_model_status_section():
    """
    Creates the model status section in the sidebar.
    """
    return html.Div(
        className="sidebar-model-status",
        children=[
            html.H5("Model Status", className="model-status-title"),
            html.Div(
                className="model-status-grid",
                children=[
                    create_model_status_item(
                        model_name="Transformer",
                        status="active",
                        accuracy=92.5
                    ),
                    create_model_status_item(
                        model_name="LSTM-Attention",
                        status="standby",
                        accuracy=89.3
                    ),
                    create_model_status_item(
                        model_name="CNN-LSTM",
                        status="training",
                        accuracy=87.8
                    ),
                    create_model_status_item(
                        model_name="Ensemble",
                        status="inactive",
                        accuracy=94.2
                    ),
                    create_model_status_item(
                        model_name="RL Model",
                        status="active",
                        accuracy=91.7
                    )
                ]
            )
        ]
    )


def create_model_status_item(model_name, status, accuracy):
    """
    Creates a single model status item.
    
    Args:
        model_name: Name of the model
        status: Current status (active, standby, training, inactive)
        accuracy: Model accuracy percentage
    
    Returns:
        html.Div: Model status item
    """
    status_colors = {
        "active": "#10b981",
        "standby": "#f59e0b",
        "training": "#3b82f6",
        "inactive": "#6b7280"
    }
    
    return html.Div(
        className="model-status-item",
        children=[
            html.Div(
                className="model-info",
                children=[
                    html.Span(model_name, className="model-name"),
                    html.Div(
                        className="model-accuracy",
                        children=[
                            html.Span(f"{accuracy}%", className="accuracy-value"),
                            html.Span("Accuracy", className="accuracy-label")
                        ]
                    )
                ]
            ),
            html.Div(
                className="model-status",
                children=[
                    html.Div(
                        className="status-dot",
                        style={"backgroundColor": status_colors.get(status, "#6b7280")}
                    ),
                    html.Span(status.upper(), className="status-text")
                ]
            )
        ]
    )


def create_system_info_section():
    """
    Creates the system information section in the sidebar.
    """
    return html.Div(
        className="sidebar-system-info",
        children=[
            html.H5("System Info", className="system-info-title"),
            html.Div(
                className="system-info-grid",
                children=[
                    create_system_info_item(
                        icon="material-symbols:memory",
                        label="CPU Usage",
                        value="24%",
                        id="cpu-usage"
                    ),
                    create_system_info_item(
                        icon="material-symbols:sd-storage",
                        label="Memory",
                        value="3.2/16GB",
                        id="memory-usage"
                    ),
                    create_system_info_item(
                        icon="material-symbols:bolt",
                        label="Latency",
                        value="12ms",
                        id="system-latency"
                    ),
                    create_system_info_item(
                        icon="material-symbols:update",
                        label="Uptime",
                        value="7d 3h",
                        id="system-uptime"
                    ),
                    create_system_info_item(
                        icon="material-symbols:database",
                        label="Data Points",
                        value="2.4M",
                        id="data-points"
                    ),
                    create_system_info_item(
                        icon="material-symbols:flash-on",
                        label="Trades Today",
                        value="42",
                        id="trades-today"
                    )
                ]
            ),
            # Connection status
            html.Div(
                className="connection-status",
                children=[
                    html.Div(
                        className="connection-item",
                        children=[
                            html.Div(
                                className="connection-dot exchange",
                                id="exchange-connection-dot"
                            ),
                            html.Span("Exchange API", className="connection-label")
                        ]
                    ),
                    html.Div(
                        className="connection-item",
                        children=[
                            html.Div(
                                className="connection-dot database",
                                id="database-connection-dot"
                            ),
                            html.Span("Database", className="connection-label")
                        ]
                    ),
                    html.Div(
                        className="connection-item",
                        children=[
                            html.Div(
                                className="connection-dot models",
                                id="models-connection-dot"
                            ),
                            html.Span("ML Models", className="connection-label")
                        ]
                    )
                ]
            )
        ]
    )


def create_system_info_item(icon, label, value, id=None):
    """
    Creates a single system information item.
    
    Args:
        icon: Icon name
        label: Item label
        value: Item value
        id: Element ID
    
    Returns:
        html.Div: System info item
    """
    return html.Div(
        className="system-info-item",
        id=id,
        children=[
            DashIconify(
                icon=icon,
                className="system-info-icon",
                width=16,
                height=16
            ),
            html.Div(
                className="system-info-content",
                children=[
                    html.Span(label, className="system-info-label"),
                    html.Span(value, className="system-info-value")
                ]
            )
        ]
    )


def create_alert_sidebar():
    """
    Creates a collapsible alerts sidebar (for notifications).
    """
    return html.Div(
        className="alerts-sidebar",
        id="alerts-sidebar",
        children=[
            html.Div(
                className="alerts-header",
                children=[
                    html.H4("Alerts & Notifications"),
                    dbc.Button(
                        DashIconify(icon="material-symbols:close", width=20),
                        id="close-alerts-btn",
                        color="link",
                        className="close-btn"
                    )
                ]
            ),
            html.Div(
                className="alerts-list",
                id="alerts-list",
                children=[
                    # Alerts will be populated dynamically
                ]
            ),
            html.Div(
                className="alerts-footer",
                children=[
                    dbc.Button(
                        "Mark All as Read",
                        id="mark-all-read-btn",
                        color="secondary",
                        size="sm"
                    ),
                    dbc.Button(
                        "Clear All",
                        id="clear-alerts-btn",
                        color="danger",
                        size="sm"
                    )
                ]
            )
        ]
    )


def create_alert_item(alert_type, message, time, is_new=True):
    """
    Creates a single alert item for the alerts sidebar.
    
    Args:
        alert_type: Type of alert (success, warning, error, info)
        message: Alert message
        time: Time of alert
        is_new: Whether the alert is new
    
    Returns:
        html.Div: Alert item
    """
    type_icons = {
        "success": "material-symbols:check-circle",
        "warning": "material-symbols:warning",
        "error": "material-symbols:error",
        "info": "material-symbols:info"
    }
    
    type_colors = {
        "success": "#10b981",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "info": "#3b82f6"
    }
    
    return html.Div(
        className=f"alert-item {'new' if is_new else ''}",
        children=[
            html.Div(
                className="alert-icon",
                children=[
                    DashIconify(
                        icon=type_icons.get(alert_type, "material-symbols:info"),
                        width=20,
                        height=20,
                        color=type_colors.get(alert_type, "#3b82f6")
                    )
                ]
            ),
            html.Div(
                className="alert-content",
                children=[
                    html.P(message, className="alert-message"),
                    html.Span(time, className="alert-time")
                ]
            ),
            html.Div(
                className="alert-actions",
                children=[
                    dbc.Button(
                        DashIconify(icon="material-symbols:close", width=16),
                        color="link",
                        size="sm",
                        className="dismiss-alert-btn"
                    ) if is_new else None
                ]
            )
        ]
    )