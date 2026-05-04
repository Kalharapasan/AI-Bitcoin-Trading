import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from .sidebars import create_sidebar

def create_main_layout():
    """
    Creates the main layout for the Bitcoin Trading AI dashboard.
    """
    return html.Div([
        # Store components for client-side data
        dcc.Store(id='session-store', storage_type='session'),
        dcc.Store(id='selected-model', data='transformer'),
        dcc.Store(id='trading-status', data={'is_running': False}),
        dcc.Store(id='alert-data', data=[]),
        
        # Main container
        dbc.Container(
            fluid=True,
            className="main-container",
            children=[
                # Header
                create_header(),
                
                # Body with sidebar and content
                dbc.Row(
                    className="dashboard-body",
                    children=[
                        # Sidebar
                        dbc.Col(
                            md=3,
                            lg=2,
                            className="sidebar-col",
                            children=create_sidebar()
                        ),
                        
                        # Main content area
                        dbc.Col(
                            md=9,
                            lg=10,
                            className="content-col",
                            children=[
                                # Tab navigation
                                dcc.Tabs(
                                    id="main-tabs",
                                    value="dashboard",
                                    className="main-tabs",
                                    children=[
                                        dcc.Tab(
                                            label="Dashboard",
                                            value="dashboard",
                                            className="main-tab",
                                            selected_className="main-tab--selected"
                                        ),
                                        dcc.Tab(
                                            label="Trading",
                                            value="trading",
                                            className="main-tab",
                                            selected_className="main-tab--selected"
                                        ),
                                        dcc.Tab(
                                            label="Models",
                                            value="models",
                                            className="main-tab",
                                            selected_className="main-tab--selected"
                                        ),
                                        dcc.Tab(
                                            label="Backtesting",
                                            value="backtesting",
                                            className="main-tab",
                                            selected_className="main-tab--selected"
                                        ),
                                        dcc.Tab(
                                            label="Monitoring",
                                            value="monitoring",
                                            className="main-tab",
                                            selected_className="main-tab--selected"
                                        ),
                                    ]
                                ),
                                
                                # Content area for tabs
                                html.Div(
                                    id="tab-content",
                                    className="tab-content"
                                ),
                                
                                # Footer
                                create_footer()
                            ]
                        )
                    ]
                )
            ]
        ),
        
        # Interval components for updates
        dcc.Interval(
            id='price-update-interval',
            interval=10*1000,  # 10 seconds
            n_intervals=0
        ),
        dcc.Interval(
            id='metrics-update-interval',
            interval=30*1000,  # 30 seconds
            n_intervals=0
        ),
        dcc.Interval(
            id='alerts-update-interval',
            interval=60*1000,  # 1 minute
            n_intervals=0
        ),
        
        # Global modals
        create_confirmation_modal(),
        create_alert_modal(),
        create_settings_modal()
    ])


def create_header():
    """
    Creates the dashboard header.
    """
    return html.Header(
        className="dashboard-header",
        children=[
            dbc.Row(
                className="align-items-center",
                children=[
                    # Logo and title
                    dbc.Col(
                        md=3,
                        children=[
                            html.Div(
                                className="header-logo",
                                children=[
                                    html.Img(
                                        src="/assets/logo.svg",
                                        className="logo-img",
                                        alt="Bitcoin Trading AI"
                                    ),
                                    html.H1(
                                        "Bitcoin Trading AI",
                                        className="header-title"
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # Live stats
                    dbc.Col(
                        md=6,
                        children=[
                            dbc.Row(
                                className="live-stats",
                                children=[
                                    dbc.Col(
                                        children=[
                                            html.Div(
                                                className="stat-item",
                                                children=[
                                                    html.Span("BTC Price:", className="stat-label"),
                                                    html.Span(
                                                        id="btc-price",
                                                        className="stat-value",
                                                        children="$0.00"
                                                    )
                                                ]
                                            )
                                        ]
                                    ),
                                    dbc.Col(
                                        children=[
                                            html.Div(
                                                className="stat-item",
                                                children=[
                                                    html.Span("24h Change:", className="stat-label"),
                                                    html.Span(
                                                        id="btc-change",
                                                        className="stat-value change-positive",
                                                        children="+0.00%"
                                                    )
                                                ]
                                            )
                                        ]
                                    ),
                                    dbc.Col(
                                        children=[
                                            html.Div(
                                                className="stat-item",
                                                children=[
                                                    html.Span("Active Model:", className="stat-label"),
                                                    html.Span(
                                                        id="active-model",
                                                        className="stat-value",
                                                        children="Transformer"
                                                    )
                                                ]
                                            )
                                        ]
                                    ),
                                    dbc.Col(
                                        children=[
                                            html.Div(
                                                className="stat-item",
                                                children=[
                                                    html.Span("Trading:", className="stat-label"),
                                                    html.Span(
                                                        id="trading-status-badge",
                                                        className="badge badge-stopped",
                                                        children="STOPPED"
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # User controls
                    dbc.Col(
                        md=3,
                        children=[
                            html.Div(
                                className="user-controls",
                                children=[
                                    # Notifications
                                    html.Button(
                                        html.I(className="fas fa-bell"),
                                        id="notifications-btn",
                                        className="header-btn",
                                        n_clicks=0
                                    ),
                                    dbc.Badge(
                                        id="notification-count",
                                        color="danger",
                                        pill=True,
                                        className="notification-badge",
                                        children="0"
                                    ),
                                    
                                    # Settings
                                    html.Button(
                                        html.I(className="fas fa-cog"),
                                        id="settings-btn",
                                        className="header-btn",
                                        n_clicks=0
                                    ),
                                    
                                    # User profile
                                    html.Div(
                                        className="user-profile",
                                        children=[
                                            html.Img(
                                                src="/assets/user-avatar.png",
                                                className="user-avatar",
                                                alt="User"
                                            ),
                                            html.Span(
                                                "Admin",
                                                className="user-name"
                                            ),
                                            html.I(
                                                className="fas fa-chevron-down",
                                                style={"marginLeft": "5px"}
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


def create_footer():
    """
    Creates the dashboard footer.
    """
    return html.Footer(
        className="dashboard-footer",
        children=[
            dbc.Row(
                className="align-items-center",
                children=[
                    dbc.Col(
                        md=6,
                        children=[
                            html.P(
                                "© 2024 Bitcoin Trading AI System. All rights reserved.",
                                className="footer-text"
                            )
                        ]
                    ),
                    dbc.Col(
                        md=6,
                        children=[
                            html.Div(
                                className="footer-links",
                                children=[
                                    html.A(
                                        "Documentation",
                                        href="#",
                                        className="footer-link"
                                    ),
                                    html.A(
                                        "API",
                                        href="#",
                                        className="footer-link"
                                    ),
                                    html.A(
                                        "Support",
                                        href="#",
                                        className="footer-link"
                                    ),
                                    html.A(
                                        "GitHub",
                                        href="#",
                                        className="footer-link"
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # System status
            html.Div(
                className="system-status",
                children=[
                    html.Span(
                        "System Status:",
                        className="status-label"
                    ),
                    html.Span(
                        id="system-status-indicator",
                        className="status-indicator status-online",
                        children="● ONLINE"
                    ),
                    html.Span(
                        id="last-update-time",
                        className="update-time",
                        children="Last update: --:--:--"
                    )
                ]
            )
        ]
    )


def create_confirmation_modal():
    """
    Creates a reusable confirmation modal.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(id="confirmation-modal-header"),
            dbc.ModalBody(id="confirmation-modal-body"),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id="confirmation-cancel-btn",
                        className="mr-2",
                        color="secondary"
                    ),
                    dbc.Button(
                        "Confirm",
                        id="confirmation-ok-btn",
                        color="primary"
                    )
                ]
            )
        ],
        id="confirmation-modal",
        centered=True,
        backdrop="static"
    )


def create_alert_modal():
    """
    Creates a modal for displaying alerts.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader("System Alerts"),
            dbc.ModalBody(id="alert-modal-body"),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id="alert-close-btn",
                    className="ml-auto"
                )
            )
        ],
        id="alert-modal",
        size="lg",
        centered=True
    )


def create_settings_modal():
    """
    Creates a modal for system settings.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader("System Settings"),
            dbc.ModalBody(
                [
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                create_trading_settings(),
                                label="Trading",
                                tab_id="trading-settings"
                            ),
                            dbc.Tab(
                                create_model_settings(),
                                label="Models",
                                tab_id="model-settings"
                            ),
                            dbc.Tab(
                                create_risk_settings(),
                                label="Risk",
                                tab_id="risk-settings"
                            )
                        ],
                        id="settings-tabs",
                        active_tab="trading-settings"
                    )
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Reset",
                        id="settings-reset-btn",
                        color="secondary",
                        className="mr-2"
                    ),
                    dbc.Button(
                        "Save",
                        id="settings-save-btn",
                        color="primary"
                    )
                ]
            )
        ],
        id="settings-modal",
        size="lg",
        centered=True
    )


def create_trading_settings():
    """
    Creates trading settings form.
    """
    return dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Trading Mode", html_for="trading-mode"),
                    dbc.Select(
                        id="trading-mode",
                        options=[
                            {"label": "Paper Trading", "value": "paper"},
                            {"label": "Live Trading", "value": "live"},
                            {"label": "Backtest Only", "value": "backtest"}
                        ],
                        value="paper"
                    )
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Trade Size (BTC)", html_for="trade-size"),
                    dbc.Input(
                        id="trade-size",
                        type="number",
                        value=0.01,
                        min=0.001,
                        step=0.001
                    )
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Max Open Positions", html_for="max-positions"),
                    dbc.Input(
                        id="max-positions",
                        type="number",
                        value=3,
                        min=1,
                        max=10
                    )
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("API Key", html_for="api-key"),
                    dbc.Input(
                        id="api-key",
                        type="password",
                        placeholder="Enter exchange API key"
                    )
                ]
            )
        ]
    )


def create_model_settings():
    """
    Creates model settings form.
    """
    return dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Primary Model", html_for="primary-model"),
                    dbc.Select(
                        id="primary-model",
                        options=[
                            {"label": "Transformer", "value": "transformer"},
                            {"label": "LSTM with Attention", "value": "lstm_attention"},
                            {"label": "CNN-LSTM", "value": "cnn_lstm"},
                            {"label": "Ensemble", "value": "ensemble"},
                            {"label": "Reinforcement Learning", "value": "rl"}
                        ],
                        value="transformer"
                    )
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Prediction Horizon", html_for="prediction-horizon"),
                    dbc.Select(
                        id="prediction-horizon",
                        options=[
                            {"label": "1 minute", "value": "1m"},
                            {"label": "5 minutes", "value": "5m"},
                            {"label": "15 minutes", "value": "15m"},
                            {"label": "1 hour", "value": "1h"},
                            {"label": "4 hours", "value": "4h"},
                            {"label": "1 day", "value": "1d"}
                        ],
                        value="15m"
                    )
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Retrain Frequency", html_for="retrain-frequency"),
                    dbc.Select(
                        id="retrain-frequency",
                        options=[
                            {"label": "Daily", "value": "daily"},
                            {"label": "Weekly", "value": "weekly"},
                            {"label": "Monthly", "value": "monthly"},
                            {"label": "When accuracy drops", "value": "adaptive"}
                        ],
                        value="weekly"
                    )
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Checklist(
                        options=[
                            {"label": "Enable Model Ensembling", "value": "ensembling"},
                            {"label": "Use Online Learning", "value": "online"},
                            {"label": "Auto-select Best Model", "value": "auto-select"}
                        ],
                        value=["ensembling", "auto-select"],
                        id="model-options"
                    )
                ]
            )
        ]
    )


def create_risk_settings():
    """
    Creates risk management settings form.
    """
    return dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Max Daily Loss (%)", html_for="max-daily-loss"),
                    dbc.Input(
                        id="max-daily-loss",
                        type="number",
                        value=5,
                        min=1,
                        max=20
                    )
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Stop Loss (%)", html_for="stop-loss"),
                    dbc.Input(
                        id="stop-loss",
                        type="number",
                        value=2,
                        min=0.5,
                        max=10,
                        step=0.5
                    )
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Take Profit (%)", html_for="take-profit"),
                    dbc.Input(
                        id="take-profit",
                        type="number",
                        value=4,
                        min=1,
                        max=20,
                        step=0.5
                    )
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Risk per Trade (%)", html_for="risk-per-trade"),
                    dbc.Input(
                        id="risk-per-trade",
                        type="number",
                        value=1,
                        min=0.1,
                        max=5,
                        step=0.1
                    )
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Position Sizing Method", html_for="position-sizing"),
                    dbc.Select(
                        id="position-sizing",
                        options=[
                            {"label": "Fixed", "value": "fixed"},
                            {"label": "Kelly Criterion", "value": "kelly"},
                            {"label": "Volatility Adjusted", "value": "volatility"},
                            {"label": "Equal Risk", "value": "equal_risk"}
                        ],
                        value="fixed"
                    )
                ]
            )
        ]
    )