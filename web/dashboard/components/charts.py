"""
Advanced charting module for Bitcoin Trading Application.
Provides comprehensive visualization tools for trading data, analysis results, and real-time monitoring.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import math

# Suppress warnings
warnings.filterwarnings('ignore')

# Import project modules
from logger import get_logger

logger = get_logger(__name__)

class ChartTheme(Enum):
    """Chart themes for different visualization styles."""
    DARK = "dark"
    LIGHT = "light"
    TRADING_VIEW = "trading_view"
    MINIMAL = "minimal"
    CUSTOM = "custom"

class ChartType(Enum):
    """Types of charts available."""
    OHLC = "ohlc"
    CANDLESTICK = "candlestick"
    LINE = "line"
    AREA = "area"
    BAR = "bar"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    CONTOUR = "contour"
    BOX = "box"
    VIOLIN = "violin"
    PIE = "pie"
    SUNBURST = "sunburst"
    TREEMAP = "treemap"
    WATERFALL = "waterfall"
    FUNNEL = "funnel"

@dataclass
class ChartConfig:
    """Configuration for charts."""
    theme: ChartTheme = ChartTheme.DARK
    width: int = 1200
    height: int = 600
    title: str = ""
    xaxis_title: str = "Date"
    yaxis_title: str = "Price"
    show_legend: bool = True
    grid_lines: bool = True
    hover_mode: str = "x unified"
    template: str = "plotly_dark"
    colors: List[str] = None
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    background_color: str = "#1e1e1e"
    paper_bgcolor: str = "#1e1e1e"
    plot_bgcolor: str = "#2d2d2d"
    
    def __post_init__(self):
        """Initialize default colors based on theme."""
        if self.colors is None:
            if self.theme == ChartTheme.DARK:
                self.colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#386FA4"]
            elif self.theme == ChartTheme.LIGHT:
                self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            elif self.theme == ChartTheme.TRADING_VIEW:
                self.colors = ["#2962FF", "#FF6B6B", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
            else:  # MINIMAL
                self.colors = ["#000000", "#666666", "#999999", "#CCCCCC", "#333333", "#555555"]
        
        # Set template based on theme
        if self.theme == ChartTheme.DARK:
            self.template = "plotly_dark"
            self.background_color = "#1e1e1e"
            self.paper_bgcolor = "#1e1e1e"
            self.plot_bgcolor = "#2d2d2d"
        elif self.theme == ChartTheme.LIGHT:
            self.template = "plotly_white"
            self.background_color = "#ffffff"
            self.paper_bgcolor = "#ffffff"
            self.plot_bgcolor = "#f5f5f5"
        elif self.theme == ChartTheme.TRADING_VIEW:
            self.template = "plotly_dark"
            self.background_color = "#131722"
            self.paper_bgcolor = "#131722"
            self.plot_bgcolor = "#1e222d"

class TradingChart:
    """
    Advanced trading chart with multiple indicators, signals, and customization options.
    """
    
    def __init__(self, config: ChartConfig = None):
        """
        Initialize trading chart.
        
        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()
        self.fig = None
        self.data = None
        self.indicators = {}
        self.signals = {}
        self.annotations = []
        self.shapes = []
        self.images = []
        
        # Set default template
        pio.templates.default = self.config.template
        
        logger.debug(f"Initialized TradingChart with theme: {self.config.theme.value}")
    
    def load_data(self, data: pd.DataFrame):
        """
        Load OHLCV data for charting.
        
        Args:
            data: DataFrame with OHLCV data and datetime index
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        self.data = data.copy()
        logger.info(f"Loaded data: {len(self.data)} candles from {self.data.index[0]} to {self.data.index[-1]}")
    
    def add_indicator(self, 
                     name: str,
                     values: pd.Series,
                     color: str = None,
                     width: int = 2,
                     style: str = "line",
                     yaxis: str = "y",
                     show_legend: bool = True,
                     **kwargs):
        """
        Add technical indicator to chart.
        
        Args:
            name: Indicator name
            values: Indicator values (must align with data index)
            color: Line color
            width: Line width
            style: Line style (line, dash, dot)
            yaxis: Y-axis to use (y, y2, y3, etc.)
            show_legend: Whether to show in legend
            **kwargs: Additional Plotly trace parameters
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Align indicator with data
        aligned_values = values.reindex(self.data.index)
        
        # Default color
        if color is None:
            color = self.config.colors[len(self.indicators) % len(self.config.colors)]
        
        # Store indicator
        self.indicators[name] = {
            'values': aligned_values,
            'color': color,
            'width': width,
            'style': style,
            'yaxis': yaxis,
            'show_legend': show_legend,
            'kwargs': kwargs
        }
        
        logger.debug(f"Added indicator: {name}")
    
    def add_signal(self,
                  name: str,
                  signals: pd.Series,
                  signal_type: str = "marker",
                  color: str = None,
                  size: int = 10,
                  symbol: str = "triangle-up",
                  text: str = None,
                  **kwargs):
        """
        Add trading signals to chart.
        
        Args:
            name: Signal name
            signals: Boolean series indicating signal points
            signal_type: Type of signal (marker, arrow, shape)
            color: Signal color
            size: Marker size
            symbol: Marker symbol
            text: Text annotation
            **kwargs: Additional parameters
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Align signals with data
        aligned_signals = signals.reindex(self.data.index)
        
        # Default colors for different signal types
        if color is None:
            if name.lower() in ['buy', 'long']:
                color = "#00FF00"  # Green
            elif name.lower() in ['sell', 'short']:
                color = "#FF0000"  # Red
            elif name.lower() == 'hold':
                color = "#FFFF00"  # Yellow
            else:
                color = self.config.colors[len(self.signals) % len(self.config.colors)]
        
        # Store signal
        self.signals[name] = {
            'signals': aligned_signals,
            'type': signal_type,
            'color': color,
            'size': size,
            'symbol': symbol,
            'text': text,
            'kwargs': kwargs
        }
        
        logger.debug(f"Added signal: {name}")
    
    def add_annotation(self,
                      text: str,
                      x: Union[datetime, str],
                      y: float,
                      xref: str = "x",
                      yref: str = "y",
                      showarrow: bool = True,
                      arrowhead: int = 2,
                      ax: int = 0,
                      ay: int = -40,
                      bgcolor: str = "rgba(255, 255, 255, 0.8)",
                      bordercolor: str = "black",
                      borderwidth: int = 1,
                      **kwargs):
        """
        Add text annotation to chart.
        
        Args:
            text: Annotation text
            x: X coordinate (datetime or string)
            y: Y coordinate
            xref: X reference ('x', 'paper', etc.)
            yref: Y reference ('y', 'paper', etc.)
            showarrow: Show arrow pointing to location
            arrowhead: Arrowhead style
            ax: Arrow x offset
            ay: Arrow y offset
            bgcolor: Background color
            bordercolor: Border color
            borderwidth: Border width
            **kwargs: Additional annotation parameters
        """
        annotation = go.layout.Annotation(
            text=text,
            x=x,
            y=y,
            xref=xref,
            yref=yref,
            showarrow=showarrow,
            arrowhead=arrowhead,
            ax=ax,
            ay=ay,
            bgcolor=bgcolor,
            bordercolor=bordercolor,
            borderwidth=borderwidth,
            **kwargs
        )
        
        self.annotations.append(annotation)
        logger.debug(f"Added annotation: {text}")
    
    def add_shape(self,
                 shape_type: str,
                 x0: Union[datetime, str, float],
                 y0: float,
                 x1: Union[datetime, str, float],
                 y1: float,
                 xref: str = "x",
                 yref: str = "y",
                 line_color: str = "white",
                 line_width: int = 1,
                 line_dash: str = "dash",
                 fillcolor: str = None,
                 opacity: float = 0.3,
                 **kwargs):
        """
        Add shape (line, rectangle, circle) to chart.
        
        Args:
            shape_type: Type of shape (line, rect, circle)
            x0: Starting x coordinate
            y0: Starting y coordinate
            x1: Ending x coordinate
            y1: Ending y coordinate
            xref: X reference
            yref: Y reference
            line_color: Line color
            line_width: Line width
            line_dash: Line dash style
            fillcolor: Fill color (for rectangles/circles)
            opacity: Opacity
            **kwargs: Additional shape parameters
        """
        shape = go.layout.Shape(
            type=shape_type,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            xref=xref,
            yref=yref,
            line=dict(color=line_color, width=line_width, dash=line_dash),
            fillcolor=fillcolor,
            opacity=opacity,
            **kwargs
        )
        
        self.shapes.append(shape)
        logger.debug(f"Added shape: {shape_type}")
    
    def create_ohlc_chart(self, 
                         show_volume: bool = True,
                         volume_height: float = 0.2,
                         volume_colors: Tuple[str, str] = ("green", "red"),
                         **kwargs) -> go.Figure:
        """
        Create OHLC or candlestick chart.
        
        Args:
            show_volume: Whether to show volume subplot
            volume_height: Height of volume subplot relative to main chart
            volume_colors: Colors for up/down volume bars
            **kwargs: Additional figure parameters
        
        Returns:
            go.Figure: Plotly figure object
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Determine subplot configuration
        if show_volume:
            row_heights = [1 - volume_height, volume_height]
            specs = [[{"secondary_y": True}], [{}]]
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=row_heights,
                specs=specs
            )
        else:
            fig = go.Figure()
        
        # Add OHLC or candlestick trace
        ohlc_trace = go.Ohlc(
            x=self.data.index,
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            name="OHLC",
            increasing_line_color=self.config.colors[0],
            decreasing_line_color=self.config.colors[1],
            **kwargs.get('ohlc_kwargs', {})
        )
        
        candlestick_trace = go.Candlestick(
            x=self.data.index,
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            name="Candlestick",
            increasing_line_color=self.config.colors[0],
            decreasing_line_color=self.config.colors[1],
            **kwargs.get('candlestick_kwargs', {})
        )
        
        # Choose which trace to add
        if kwargs.get('chart_type', 'candlestick') == 'ohlc':
            trace = ohlc_trace
        else:
            trace = candlestick_trace
        
        if show_volume:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)
        
        # Add volume if requested
        if show_volume:
            # Calculate colors for volume bars
            volume_colors_list = []
            for i in range(len(self.data)):
                if i == 0:
                    volume_colors_list.append(volume_colors[0])
                else:
                    if self.data['close'].iloc[i] >= self.data['close'].iloc[i-1]:
                        volume_colors_list.append(volume_colors[0])
                    else:
                        volume_colors_list.append(volume_colors[1])
            
            volume_trace = go.Bar(
                x=self.data.index,
                y=self.data['volume'],
                name="Volume",
                marker_color=volume_colors_list,
                opacity=0.7,
                **kwargs.get('volume_kwargs', {})
            )
            
            fig.add_trace(volume_trace, row=2, col=1)
            
            # Update volume subplot layout
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Add indicators
        for name, indicator in self.indicators.items():
            if indicator['yaxis'].startswith('y') and indicator['yaxis'] != 'y2':
                # Main y-axis indicators
                trace = go.Scatter(
                    x=self.data.index,
                    y=indicator['values'],
                    name=name,
                    line=dict(
                        color=indicator['color'],
                        width=indicator['width'],
                        dash=indicator['style']
                    ),
                    mode='lines',
                    showlegend=indicator['show_legend'],
                    **indicator['kwargs']
                )
                
                if show_volume:
                    fig.add_trace(trace, row=1, col=1)
                else:
                    fig.add_trace(trace)
        
        # Add signals
        for name, signal in self.signals.items():
            signal_points = self.data[signal['signals'].fillna(False)]
            
            if len(signal_points) > 0:
                if signal['type'] == 'marker':
                    trace = go.Scatter(
                        x=signal_points.index,
                        y=signal_points['close'],
                        name=name,
                        mode='markers',
                        marker=dict(
                            color=signal['color'],
                            size=signal['size'],
                            symbol=signal['symbol'],
                            line=dict(width=1, color='white')
                        ),
                        text=signal['text'],
                        **signal['kwargs']
                    )
                    
                    if show_volume:
                        fig.add_trace(trace, row=1, col=1)
                    else:
                        fig.add_trace(trace)
                
                elif signal['type'] == 'arrow':
                    # Add arrows as annotations
                    for idx, row in signal_points.iterrows():
                        arrow_symbol = "▲" if 'buy' in name.lower() else "▼"
                        self.add_annotation(
                            text=arrow_symbol,
                            x=idx,
                            y=row['high'] * 1.02,
                            showarrow=False,
                            font=dict(size=20, color=signal['color'])
                        )
        
        # Update layout
        fig.update_layout(
            title=self.config.title or f"Price Chart - {self.data.index[0].date()} to {self.data.index[-1].date()}",
            xaxis_title=self.config.xaxis_title,
            yaxis_title=self.config.yaxis_title,
            showlegend=self.config.show_legend,
            hovermode=self.config.hover_mode,
            template=self.config.template,
            font=dict(
                family=self.config.font_family,
                size=self.config.font_size
            ),
            plot_bgcolor=self.config.plot_bgcolor,
            paper_bgcolor=self.config.paper_bgcolor,
            height=self.config.height,
            width=self.config.width,
            annotations=self.annotations,
            shapes=self.shapes
        )
        
        # Update axes
        fig.update_xaxes(
            rangeslider_visible=kwargs.get('rangeslider', False),
            rangeselector=kwargs.get('rangeselector', None)
        )
        
        if show_volume:
            fig.update_yaxes(title_text="Price", row=1, col=1)
        
        self.fig = fig
        logger.info(f"Created OHLC chart with {len(self.indicators)} indicators and {len(self.signals)} signals")
        
        return fig
    
    def create_equity_curve_chart(self,
                                 equity_data: pd.DataFrame,
                                 benchmark_data: Optional[pd.DataFrame] = None,
                                 show_drawdown: bool = True,
                                 **kwargs) -> go.Figure:
        """
        Create equity curve chart with drawdown.
        
        Args:
            equity_data: DataFrame with equity curve (must have 'equity' column)
            benchmark_data: Optional benchmark data for comparison
            show_drawdown: Whether to show drawdown subplot
            **kwargs: Additional figure parameters
        
        Returns:
            go.Figure: Plotly figure object
        """
        if 'equity' not in equity_data.columns:
            raise ValueError("equity_data must have 'equity' column")
        
        # Determine subplot configuration
        if show_drawdown:
            row_heights = [0.7, 0.3]
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=row_heights,
                subplot_titles=("Equity Curve", "Drawdown")
            )
        else:
            fig = go.Figure()
        
        # Ensure datetime index
        if not isinstance(equity_data.index, pd.DatetimeIndex):
            equity_data.index = pd.to_datetime(equity_data.index)
        
        # Add equity curve
        equity_trace = go.Scatter(
            x=equity_data.index,
            y=equity_data['equity'],
            name="Portfolio",
            line=dict(color=self.config.colors[0], width=2),
            fill='tozeroy' if kwargs.get('fill_area', False) else None,
            fillcolor='rgba(46, 134, 171, 0.2)' if kwargs.get('fill_area', False) else None
        )
        
        if show_drawdown:
            fig.add_trace(equity_trace, row=1, col=1)
        else:
            fig.add_trace(equity_trace)
        
        # Add benchmark if provided
        if benchmark_data is not None:
            if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                benchmark_data.index = pd.to_datetime(benchmark_data.index)
            
            # Align benchmark with equity data
            aligned_benchmark = benchmark_data.reindex(equity_data.index).ffill()
            
            if 'value' in aligned_benchmark.columns:
                benchmark_trace = go.Scatter(
                    x=aligned_benchmark.index,
                    y=aligned_benchmark['value'],
                    name="Benchmark",
                    line=dict(color=self.config.colors[1], width=2, dash='dash')
                )
                
                if show_drawdown:
                    fig.add_trace(benchmark_trace, row=1, col=1)
                else:
                    fig.add_trace(benchmark_trace)
        
        # Add drawdown subplot
        if show_drawdown:
            # Calculate drawdown
            equity_series = equity_data['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            
            drawdown_trace = go.Scatter(
                x=equity_data.index,
                y=drawdown,
                name="Drawdown",
                line=dict(color='red', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)'
            )
            
            fig.add_trace(drawdown_trace, row=2, col=1)
            
            # Add horizontal line at 0
            fig.add_shape(
                type="line",
                x0=equity_data.index[0],
                y0=0,
                x1=equity_data.index[-1],
                y1=0,
                line=dict(color="white", width=1, dash="dash"),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=self.config.title or "Equity Curve Analysis",
            xaxis_title=self.config.xaxis_title,
            showlegend=self.config.show_legend,
            hovermode=self.config.hover_mode,
            template=self.config.template,
            font=dict(
                family=self.config.font_family,
                size=self.config.font_size
            ),
            plot_bgcolor=self.config.plot_bgcolor,
            paper_bgcolor=self.config.paper_bgcolor,
            height=self.config.height,
            width=self.config.width
        )
        
        if show_drawdown:
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        else:
            fig.update_yaxes(title_text="Portfolio Value ($)")
        
        self.fig = fig
        logger.info(f"Created equity curve chart with drawdown analysis")
        
        return fig
    
    def create_performance_metrics_chart(self,
                                        metrics: Dict[str, float],
                                        chart_type: str = "radar",
                                        **kwargs) -> go.Figure:
        """
        Create performance metrics visualization.
        
        Args:
            metrics: Dictionary of metric names to values
            chart_type: Type of chart (radar, bar, gauge)
            **kwargs: Additional figure parameters
        
        Returns:
            go.Figure: Plotly figure object
        """
        if chart_type == "radar":
            fig = self._create_radar_chart(metrics, **kwargs)
        elif chart_type == "bar":
            fig = self._create_bar_chart(metrics, **kwargs)
        elif chart_type == "gauge":
            fig = self._create_gauge_chart(metrics, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        self.fig = fig
        return fig
    
    def _create_radar_chart(self, metrics: Dict[str, float], **kwargs) -> go.Figure:
        """Create radar chart for performance metrics."""
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Normalize values for radar chart (0-100 scale)
        normalized_values = []
        for i, (cat, val) in enumerate(zip(categories, values)):
            if cat.lower() in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_factor']:
                # These are ratios, scale appropriately
                normalized = min(100, max(0, val * 20))  # Scale factor
            elif cat.lower() in ['win_rate', 'success_rate']:
                # Already in percentage
                normalized = min(100, max(0, val))
            elif cat.lower() in ['max_drawdown']:
                # Negative is bad, invert
                normalized = max(0, min(100, 100 + val))  # Drawdown is negative
            else:
                # Default scaling
                normalized = min(100, max(0, (val - min(values)) / (max(values) - min(values)) * 100))
            
            normalized_values.append(normalized)
        
        # Close the polygon
        categories = categories + [categories[0]]
        normalized_values = normalized_values + [normalized_values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=categories,
            fill='toself',
            name="Performance",
            line=dict(color=self.config.colors[0], width=2),
            fillcolor='rgba(46, 134, 171, 0.5)'
        ))
        
        # Add reference lines for ideal values
        reference_values = []
        for cat in categories[:-1]:  # Exclude the duplicate first category
            if cat.lower() in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
                reference_values.append(100)  # Ideal: 5.0 * 20 = 100
            elif cat.lower() in ['win_rate', 'success_rate']:
                reference_values.append(80)  # Ideal: 80%
            elif cat.lower() in ['max_drawdown']:
                reference_values.append(100)  # Ideal: 0% drawdown
            else:
                reference_values.append(100)
        
        reference_values = reference_values + [reference_values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=reference_values,
            theta=categories,
            name="Ideal",
            line=dict(color='green', width=1, dash='dash'),
            opacity=0.5
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11),
                    rotation=90
                )
            ),
            title=self.config.title or "Performance Metrics Radar",
            showlegend=True,
            template=self.config.template,
            height=self.config.height,
            width=self.config.width
        )
        
        return fig
    
    def _create_bar_chart(self, metrics: Dict[str, float], **kwargs) -> go.Figure:
        """Create bar chart for performance metrics."""
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Color coding based on metric type
        colors = []
        for cat in categories:
            if cat.lower() in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_factor']:
                colors.append(self.config.colors[0])  # Blue for ratios
            elif cat.lower() in ['win_rate', 'success_rate']:
                colors.append(self.config.colors[1])  # Purple for rates
            elif cat.lower() in ['max_drawdown', 'volatility']:
                colors.append(self.config.colors[2])  # Orange for risk metrics
            elif cat.lower() in ['total_return', 'annual_return']:
                colors.append(self.config.colors[3])  # Green for returns
            else:
                colors.append(self.config.colors[4])  # Default
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition='auto',
            name="Metrics"
        ))
        
        # Add horizontal reference lines
        for i, (cat, val) in enumerate(zip(categories, values)):
            if cat.lower() in ['sharpe_ratio']:
                # Add reference line at 1.0 (good Sharpe)
                fig.add_shape(
                    type="line",
                    x0=i-0.4, x1=i+0.4,
                    y0=1.0, y1=1.0,
                    line=dict(color="green", width=2, dash="dash"),
                    opacity=0.7
                )
            elif cat.lower() in ['win_rate']:
                # Add reference line at 50%
                fig.add_shape(
                    type="line",
                    x0=i-0.4, x1=i+0.4,
                    y0=50, y1=50,
                    line=dict(color="yellow", width=2, dash="dash"),
                    opacity=0.7
                )
        
        fig.update_layout(
            title=self.config.title or "Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            showlegend=False,
            template=self.config.template,
            height=self.config.height,
            width=self.config.width,
            plot_bgcolor=self.config.plot_bgcolor,
            paper_bgcolor=self.config.paper_bgcolor
        )
        
        return fig
    
    def _create_gauge_chart(self, metrics: Dict[str, float], **kwargs) -> go.Figure:
        """Create gauge chart for key metrics."""
        # Select key metrics for gauges
        key_metrics = {}
        for name, value in metrics.items():
            if name.lower() in ['sharpe_ratio', 'win_rate', 'max_drawdown', 'profit_factor']:
                key_metrics[name] = value
        
        if not key_metrics:
            key_metrics = dict(list(metrics.items())[:4])  # Take first 4
        
        # Create subplot grid
        n_metrics = len(key_metrics)
        n_cols = min(2, n_metrics)
        n_rows = math.ceil(n_metrics / n_cols)
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            specs=[[{'type': 'indicator'} for _ in range(n_cols)] for _ in range(n_rows)],
            subplot_titles=list(key_metrics.keys())
        )
        
        row = 1
        col = 1
        for i, (name, value) in enumerate(key_metrics.items()):
            # Determine gauge parameters based on metric
            if name.lower() == 'sharpe_ratio':
                min_val = -1
                max_val = 3
                thresholds = {
                    'steps': [
                        {'range': [-1, 0], 'color': "red"},
                        {'range': [0, 1], 'color': "yellow"},
                        {'range': [1, 3], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                }
            elif name.lower() == 'win_rate':
                min_val = 0
                max_val = 100
                thresholds = {
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                }
            elif name.lower() == 'max_drawdown':
                min_val = -50
                max_val = 0
                thresholds = {
                    'steps': [
                        {'range': [-50, -20], 'color': "red"},
                        {'range': [-20, -10], 'color': "yellow"},
                        {'range': [-10, 0], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                }
            else:
                min_val = min(0, value * 0.5)
                max_val = max(2, value * 1.5)
                thresholds = {
                    'steps': [
                        {'range': [min_val, min_val + (max_val-min_val)/3], 'color': "red"},
                        {'range': [min_val + (max_val-min_val)/3, min_val + 2*(max_val-min_val)/3], 'color': "yellow"},
                        {'range': [min_val + 2*(max_val-min_val)/3, max_val], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                }
            
            gauge = go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': name.replace('_', ' ').title()},
                gauge=dict(
                    axis={'range': [min_val, max_val]},
                    bar={'color': "darkblue"},
                    **thresholds
                ),
                domain={'row': row-1, 'column': col-1}
            )
            
            fig.add_trace(gauge, row=row, col=col)
            
            # Update row/col for next gauge
            col += 1
            if col > n_cols:
                col = 1
                row += 1
        
        fig.update_layout(
            title=self.config.title or "Key Performance Indicators",
            template=self.config.template,
            height=self.config.height,
            width=self.config.width,
            font=dict(size=10)
        )
        
        return fig
    
    def create_correlation_heatmap(self,
                                  data: pd.DataFrame,
                                  method: str = 'pearson',
                                  **kwargs) -> go.Figure:
        """
        Create correlation heatmap for multiple assets or indicators.
        
        Args:
            data: DataFrame with columns to correlate
            method: Correlation method (pearson, spearman, kendall)
            **kwargs: Additional figure parameters
        
        Returns:
            go.Figure: Plotly figure object
        """
        # Calculate correlation matrix
        corr_matrix = data.corr(method=method)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale=kwargs.get('colorscale', 'RdBu'),
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        # Add annotations for values
        annotations = []
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                annotations.append(
                    dict(
                        x=col,
                        y=row,
                        text=str(round(corr_matrix.iloc[i, j], 2)),
                        showarrow=False,
                        font=dict(
                            color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                            size=10
                        )
                    )
                )
        
        fig.update_layout(
            title=self.config.title or f"Correlation Heatmap ({method.capitalize()})",
            xaxis_title="Variable",
            yaxis_title="Variable",
            template=self.config.template,
            height=self.config.height,
            width=self.config.width,
            annotations=annotations
        )
        
        self.fig = fig
        return fig
    
    def create_returns_distribution(self,
                                   returns: pd.Series,
                                   benchmark_returns: Optional[pd.Series] = None,
                                   **kwargs) -> go.Figure:
        """
        Create returns distribution chart with statistics.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
            **kwargs: Additional figure parameters
        
        Returns:
            go.Figure: Plotly figure object
        """
        # Create subplot with histogram and box plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Returns Distribution", "Box Plot", "Q-Q Plot", "Statistics"),
            specs=[
                [{"colspan": 2}, None],
                [{}, {}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Histogram with KDE
        hist_trace = go.Histogram(
            x=returns,
            nbinsx=kwargs.get('bins', 50),
            name="Returns",
            marker_color=self.config.colors[0],
            opacity=0.7,
            histnorm='probability density'
        )
        
        fig.add_trace(hist_trace, row=1, col=1)
        
        # Add KDE curve
        if kwargs.get('show_kde', True):
            from scipy.stats import gaussian_kde
            
            kde = gaussian_kde(returns.dropna())
            x_range = np.linspace(returns.min(), returns.max(), 1000)
            kde_trace = go.Scatter(
                x=x_range,
                y=kde(x_range),
                name="KDE",
                line=dict(color=self.config.colors[1], width=2),
                mode='lines'
            )
            
            fig.add_trace(kde_trace, row=1, col=1)
        
        # Add normal distribution for comparison
        if kwargs.get('show_normal', True):
            from scipy.stats import norm
            
            mu, std = returns.mean(), returns.std()
            normal_trace = go.Scatter(
                x=x_range,
                y=norm.pdf(x_range, mu, std),
                name="Normal",
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.7
            )
            
            fig.add_trace(normal_trace, row=1, col=1)
        
        # 2. Box plot
        box_trace = go.Box(
            y=returns,
            name="Returns",
            marker_color=self.config.colors[0],
            boxpoints=kwargs.get('box_points', 'outliers')
        )
        
        fig.add_trace(box_trace, row=2, col=1)
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_box = go.Box(
                y=benchmark_returns,
                name="Benchmark",
                marker_color=self.config.colors[1]
            )
            
            fig.add_trace(benchmark_box, row=2, col=1)
        
        # 3. Q-Q Plot
        if kwargs.get('show_qq', True):
            from scipy.stats import probplot
            
            qq_data = probplot(returns.dropna(), dist="norm")
            qq_trace = go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name="Q-Q Plot",
                marker=dict(color=self.config.colors[0], size=6)
            )
            
            # Add reference line
            x_line = np.array([qq_data[0][0].min(), qq_data[0][0].max()])
            y_line = qq_data[1][0] + qq_data[1][1] * x_line
            line_trace = go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name="Normal Line",
                line=dict(color='red', dash='dash')
            )
            
            fig.add_trace(qq_trace, row=2, col=2)
            fig.add_trace(line_trace, row=2, col=2)
        
        # 4. Statistics table
        stats = {
            'Mean': f"{returns.mean():.4%}",
            'Std Dev': f"{returns.std():.4%}",
            'Skewness': f"{returns.skew():.4f}",
            'Kurtosis': f"{returns.kurtosis():.4f}",
            'Sharpe': f"{returns.mean()/returns.std() * np.sqrt(252):.2f}",
            'Min': f"{returns.min():.4%}",
            'Max': f"{returns.max():.4%}"
        }
        
        # Create table trace
        table_trace = go.Table(
            header=dict(
                values=['Statistic', 'Value'],
                fill_color=self.config.plot_bgcolor,
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[list(stats.keys()), list(stats.values())],
                fill_color=[self.config.paper_bgcolor],
                font=dict(color='white', size=11)
            ),
            domain=dict(x=[0, 1], y=[0, 1])
        )
        
        # Add table as annotation in fourth subplot
        fig.add_trace(table_trace, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=self.config.title or "Returns Distribution Analysis",
            template=self.config.template,
            height=self.config.height,
            width=self.config.width,
            showlegend=True,
            bargap=0.1
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Return", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Return", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        
        self.fig = fig
        return fig
    
    def create_trade_analysis_chart(self,
                                  trades: pd.DataFrame,
                                  **kwargs) -> go.Figure:
        """
        Create comprehensive trade analysis chart.
        
        Args:
            trades: DataFrame with trade information
            **kwargs: Additional figure parameters
        
        Returns:
            go.Figure: Plotly figure object
        """
        required_cols = ['entry_time', 'exit_time', 'pnl', 'side']
        for col in required_cols:
            if col not in trades.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Cumulative P&L", "Trade Duration", "P&L Distribution", "Win/Loss Analysis"),
            specs=[[{}, {}], [{}, {}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Cumulative P&L
        trades_sorted = trades.sort_values('exit_time')
        cumulative_pnl = trades_sorted['pnl'].cumsum()
        
        cum_trace = go.Scatter(
            x=trades_sorted['exit_time'],
            y=cumulative_pnl,
            mode='lines+markers',
            name="Cumulative P&L",
            line=dict(color=self.config.colors[0], width=2),
            marker=dict(
                size=8,
                color=['green' if pnl >= 0 else 'red' for pnl in trades_sorted['pnl']],
                symbol='circle'
            )
        )
        
        fig.add_trace(cum_trace, row=1, col=1)
        
        # Add horizontal line at 0
        fig.add_shape(
            type="line",
            x0=trades_sorted['exit_time'].min(),
            y0=0,
            x1=trades_sorted['exit_time'].max(),
            y1=0,
            line=dict(color="white", width=1, dash="dash"),
            row=1, col=1
        )
        
        # 2. Trade duration histogram
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            trade_durations = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600  # Convert to hours
            
            duration_trace = go.Histogram(
                x=trade_durations,
                nbinsx=20,
                name="Trade Duration",
                marker_color=self.config.colors[1],
                opacity=0.7
            )
            
            fig.add_trace(duration_trace, row=1, col=2)
        
        # 3. P&L distribution
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] <= 0]
        
        # Create grouped histogram
        fig.add_trace(go.Histogram(
            x=winning_trades['pnl'],
            name="Winning Trades",
            marker_color='green',
            opacity=0.7,
            nbinsx=20
        ), row=2, col=1)
        
        fig.add_trace(go.Histogram(
            x=losing_trades['pnl'],
            name="Losing Trades",
            marker_color='red',
            opacity=0.7,
            nbinsx=20
        ), row=2, col=1)
        
        # Add vertical line at 0
        fig.add_shape(
            type="line",
            x0=0, x1=0,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="white", width=2),
            row=2, col=1
        )
        
        # 4. Win/Loss analysis (pie chart)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        pie_trace = go.Pie(
            labels=['Winning Trades', 'Losing Trades'],
            values=[win_count, loss_count],
            marker=dict(colors=['green', 'red']),
            hole=0.4,
            textinfo='label+percent+value',
            hoverinfo='label+percent+value'
        )
        
        fig.add_trace(pie_trace, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=self.config.title or "Trade Analysis Dashboard",
            template=self.config.template,
            height=self.config.height,
            width=self.config.width,
            showlegend=True,
            barmode='overlay'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
        fig.update_xaxes(title_text="Duration (hours)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="P&L ($)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        self.fig = fig
        return fig
    
    def create_risk_metrics_chart(self,
                                risk_metrics: Dict[str, float],
                                **kwargs) -> go.Figure:
        """
        Create risk metrics visualization.
        
        Args:
            risk_metrics: Dictionary of risk metrics
            **kwargs: Additional figure parameters
        
        Returns:
            go.Figure: Plotly figure object
        """
        # Create gauge charts for key risk metrics
        key_risk_metrics = {
            'Value at Risk (95%)': risk_metrics.get('var_95', 0),
            'Conditional VaR (95%)': risk_metrics.get('cvar_95', 0),
            'Max Drawdown': risk_metrics.get('max_drawdown', 0),
            'Volatility': risk_metrics.get('volatility', 0)
        }
        
        # Filter out None values
        key_risk_metrics = {k: v for k, v in key_risk_metrics.items() if v is not None}
        
        # Create subplot grid
        n_metrics = len(key_risk_metrics)
        fig = make_subplots(
            rows=1, cols=n_metrics,
            specs=[[{'type': 'indicator'} for _ in range(n_metrics)]],
            subplot_titles=list(key_risk_metrics.keys())
        )
        
        for i, (name, value) in enumerate(key_risk_metrics.items(), 1):
            # Determine color based on risk level
            if 'VaR' in name or 'Drawdown' in name:
                # Negative values are bad (more risk)
                if value > -5:
                    color = "green"
                elif value > -10:
                    color = "yellow"
                else:
                    color = "red"
            else:
                # For volatility, moderate values are better
                if value < 0.1:
                    color = "green"
                elif value < 0.2:
                    color = "yellow"
                else:
                    color = "red"
            
            indicator = go.Indicator(
                mode="number+gauge",
                value=value,
                title={'text': name, 'font': {'size': 14}},
                number={'suffix': '%', 'font': {'size': 20, 'color': color}},
                gauge={
                    'axis': {'range': [None, 0] if value < 0 else [0, None]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [-20, -10], 'color': "red"},
                        {'range': [-10, -5], 'color': "yellow"},
                        {'range': [-5, 0], 'color': "green"}
                    ] if value < 0 else [
                        {'range': [0, 10], 'color': "green"},
                        {'range': [10, 20], 'color': "yellow"},
                        {'range': [20, 30], 'color': "red"}
                    ]
                },
                domain={'row': 0, 'column': i-1}
            )
            
            fig.add_trace(indicator, row=1, col=i)
        
        fig.update_layout(
            title=self.config.title or "Risk Metrics Dashboard",
            template=self.config.template,
            height=400,
            width=self.config.width,
            margin=dict(t=100, b=50)
        )
        
        self.fig = fig
        return fig
    
    def save_chart(self, filepath: str, format: str = 'html'):
        """
        Save chart to file.
        
        Args:
            filepath: Path to save file
            format: File format (html, png, jpeg, svg, pdf)
        """
        if self.fig is None:
            raise ValueError("No chart created. Call a create method first.")
        
        if format == 'html':
            self.fig.write_html(filepath)
        elif format in ['png', 'jpeg', 'svg', 'pdf', 'webp']:
            self.fig.write_image(filepath, format=format)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved chart to {filepath} ({format})")
    
    def get_html(self) -> str:
        """
        Get chart as HTML string.
        
        Returns:
            str: HTML string
        """
        if self.fig is None:
            raise ValueError("No chart created. Call a create method first.")
        
        return self.fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    def show(self):
        """Display chart (for Jupyter notebooks)."""
        if self.fig is None:
            raise ValueError("No chart created. Call a create method first.")
        
        self.fig.show()

# Utility functions
def create_price_chart(data: pd.DataFrame,
                      indicators: Dict[str, pd.Series] = None,
                      signals: Dict[str, pd.Series] = None,
                      config: ChartConfig = None,
                      **kwargs) -> go.Figure:
    """
    Quick function to create price chart with indicators and signals.
    
    Args:
        data: OHLCV DataFrame
        indicators: Dictionary of indicator names to Series
        signals: Dictionary of signal names to boolean Series
        config: Chart configuration
        **kwargs: Additional chart parameters
    
    Returns:
        go.Figure: Plotly figure
    """
    chart = TradingChart(config)
    chart.load_data(data)
    
    if indicators:
        for name, values in indicators.items():
            chart.add_indicator(name, values)
    
    if signals:
        for name, signal in signals.items():
            chart.add_signal(name, signal)
    
    return chart.create_ohlc_chart(**kwargs)

def create_performance_dashboard(equity_data: pd.DataFrame,
                                trades: pd.DataFrame,
                                metrics: Dict[str, float],
                                config: ChartConfig = None) -> go.Figure:
    """
    Create comprehensive performance dashboard.
    
    Args:
        equity_data: DataFrame with equity curve
        trades: DataFrame with trade history
        metrics: Performance metrics dictionary
        config: Chart configuration
    
    Returns:
        go.Figure: Plotly figure
    """
    # Create subplot grid
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Equity Curve", "Drawdown",
            "Trade P&L Distribution", "Win/Loss Analysis",
            "Performance Metrics", "Risk Metrics"
        ),
        specs=[
            [{"secondary_y": True}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "indicator"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.15
    )
    
    # 1. Equity Curve
    equity_trace = go.Scatter(
        x=equity_data.index,
        y=equity_data['equity'],
        name="Equity",
        line=dict(color="#2E86AB", width=2)
    )
    fig.add_trace(equity_trace, row=1, col=1)
    
    # 2. Drawdown
    rolling_max = equity_data['equity'].expanding().max()
    drawdown = (equity_data['equity'] - rolling_max) / rolling_max * 100
    
    drawdown_trace = go.Scatter(
        x=equity_data.index,
        y=drawdown,
        name="Drawdown",
        line=dict(color="red", width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.3)'
    )
    fig.add_trace(drawdown_trace, row=1, col=2)
    
    # 3. Trade P&L Distribution
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] <= 0]
    
    fig.add_trace(go.Histogram(
        x=winning_trades['pnl'],
        name="Winning Trades",
        marker_color='green',
        opacity=0.7,
        nbinsx=20
    ), row=2, col=1)
    
    fig.add_trace(go.Histogram(
        x=losing_trades['pnl'],
        name="Losing Trades",
        marker_color='red',
        opacity=0.7,
        nbinsx=20
    ), row=2, col=1)
    
    # 4. Win/Loss Analysis
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    fig.add_trace(go.Pie(
        labels=['Wins', 'Losses'],
        values=[win_count, loss_count],
        marker=dict(colors=['green', 'red']),
        hole=0.4,
        textinfo='label+percent+value'
    ), row=2, col=2)
    
    # 5. Performance Metrics (Bar chart)
    perf_metrics = {k: v for k, v in metrics.items() 
                   if k in ['sharpe_ratio', 'sortino_ratio', 'profit_factor', 'win_rate']}
    
    if perf_metrics:
        fig.add_trace(go.Bar(
            x=list(perf_metrics.keys()),
            y=list(perf_metrics.values()),
            marker_color='#2E86AB'
        ), row=3, col=1)
    
    # 6. Risk Metrics (Gauge)
    risk_value = metrics.get('max_drawdown', 0)
    risk_color = "green" if risk_value > -10 else "yellow" if risk_value > -20 else "red"
    
    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=risk_value,
        title={'text': "Max Drawdown", 'font': {'size': 16}},
        number={'suffix': '%', 'font': {'size': 24, 'color': risk_color}},
        gauge={
            'axis': {'range': [-30, 0]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [-30, -20], 'color': "red"},
                {'range': [-20, -10], 'color': "yellow"},
                {'range': [-10, 0], 'color': "green"}
            ]
        }
    ), row=3, col=2)
    
    # Update layout
    fig.update_layout(
        title="Performance Dashboard",
        template="plotly_dark",
        height=900,
        width=1200,
        showlegend=True,
        barmode='overlay'
    )
    
    # Update axes labels
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
    fig.update_xaxes(title_text="P&L ($)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    return fig

# Example usage
if __name__ == "__main__":
    print("Testing Charts Module...")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 50000 + np.cumsum(np.random.randn(100) * 1000)
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(100) * 50,
        'high': prices + abs(np.random.randn(100) * 100),
        'low': prices - abs(np.random.randn(100) * 100),
        'close': prices,
        'volume': np.random.rand(100) * 1000
    }, index=dates)
    
    # Generate sample indicators
    sma_20 = data['close'].rolling(window=20).mean()
    sma_50 = data['close'].rolling(window=50).mean()
    
    # Generate sample signals
    buy_signals = (sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1))
    sell_signals = (sma_20 < sma_50) & (sma_20.shift(1) >= sma_50.shift(1))
    
    # Create chart configuration
    config = ChartConfig(
        theme=ChartTheme.DARK,
        title="BTC/USDT Price Chart with Moving Averages",
        width=1400,
        height=800
    )
    
    # Create price chart
    print("Creating price chart...")
    chart = TradingChart(config)
    chart.load_data(data)
    chart.add_indicator("SMA 20", sma_20, color="#FF6B6B")
    chart.add_indicator("SMA 50", sma_50, color="#4CAF50")
    chart.add_signal("Buy", buy_signals, signal_type="marker", color="#00FF00", symbol="triangle-up")
    chart.add_signal("Sell", sell_signals, signal_type="marker", color="#FF0000", symbol="triangle-down")
    
    fig = chart.create_ohlc_chart(show_volume=True)
    chart.save_chart("price_chart.html")
    
    # Create equity curve chart
    print("Creating equity curve chart...")
    equity_data = pd.DataFrame({
        'equity': 10000 * np.cumprod(1 + np.random.randn(100) * 0.01)
    }, index=dates)
    
    equity_chart = TradingChart(config)
    equity_fig = equity_chart.create_equity_curve_chart(equity_data, show_drawdown=True)
    equity_chart.save_chart("equity_chart.html")
    
    # Create performance metrics chart
    print("Creating performance metrics chart...")
    metrics = {
        'Sharpe Ratio': 1.8,
        'Sortino Ratio': 2.1,
        'Calmar Ratio': 1.5,
        'Max Drawdown': -8.7,
        'Win Rate': 62.5,
        'Profit Factor': 1.8
    }
    
    metrics_chart = TradingChart(config)
    metrics_fig = metrics_chart.create_performance_metrics_chart(metrics, chart_type="radar")
    metrics_chart.save_chart("metrics_chart.html")
    
    # Create returns distribution chart
    print("Creating returns distribution chart...")
    returns = pd.Series(np.random.randn(1000) * 0.02, 
                       index=pd.date_range('2023-01-01', periods=1000, freq='D'))
    
    returns_chart = TradingChart(config)
    returns_fig = returns_chart.create_returns_distribution(returns)
    returns_chart.save_chart("returns_chart.html")
    
    print("\nCharts created successfully!")
    print("Saved files:")
    print("  • price_chart.html")
    print("  • equity_chart.html")
    print("  • metrics_chart.html")
    print("  • returns_chart.html")