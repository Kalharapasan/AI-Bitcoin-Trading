import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
import './App.css';

const API_BASE = 'http://localhost:8000';
function App() {
    const [prediction, setPrediction] = useState({
        current: null,
        predicted: null,
        signal: null,
        loading: false,
        error: null,
    });

    const [chartData, setChartData] = useState(null);
    const [chartLoading, setChartLoading] = useState(true);

    return (
        <div className="app">
            <header className="app-header">
                <h1>Bitcoin Trading AI Dashboard</h1>
                <p className="disclaimer">
                    Predictions are probabilistic and <strong>not</strong> financial advice.
                </p>
            </header>

            <main className="app-main">
                {/* Prediction Card */}
                <section className="card prediction-card">
                    <h2>Price Prediction</h2>
                    {prediction.loading ? (
                        <p className="status-text">Loading prediction...</p>
                    ) : prediction.error ? (
                        <p className="error-text">Error: {prediction.error}</p>
                    ) : (
                        <>
                            <div className="price-info">
                                <span className="label">Current Price:</span>
                                <span className="value">${prediction.current?.toFixed(2)}</span>
                            </div>
                            <div className="price-info">
                                <span className="label">7-Day Predicted:</span>
                                <span className="value">${prediction.predicted?.toFixed(2)}</span>
                            </div>
                            <div className="price-info">
                                <span className="label">Signal:</span>
                                <span className={getSignalClass(prediction.signal)}>
                                    {prediction.signal}
                                </span>
                            </div>
                            <button onClick={fetchPrediction} className="refresh-btn">
                                Refresh Prediction
                            </button>
                        </>
                    )}
                </section>

                {/* Price Chart */}
                <section className="card chart-card">
                    <h2>Price Trend (Last 30 Days)</h2>
                    {chartLoading ? (
                        <p className="status-text">Loading chart data...</p>
                    ) : chartData ? (
                        <div className="chart-container">
                            <Line
                                data={chartData}
                                options={{
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    plugins: {
                                        legend: { position: 'top' },
                                        tooltip: { mode: 'index', intersect: false },
                                    },
                                    scales: {
                                        y: {
                                            beginAtZero: false,
                                            ticks: {
                                                callback: (value) => `$${value.toLocaleString()}`,
                                            },
                                        },
                                    },
                                }}
                            />
                        </div>
                    ) : (
                        <p className="status-text">Failed to load chart data.</p>
                    )}
                </section>
            </main>

            <footer className="app-footer">
                <small>Powered by FastAPI &bull; React &bull; LSTM Model &bull; Data via yfinance</small>
            </footer>
        </div>
    );
}

export default App;