"""
CRUD (Create, Read, Update, Delete) operations for the Bitcoin Trading AI application.
Database operations using SQLAlchemy ORM.
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
import logging
from . import models

logger = logging.getLogger(__name__)


class CRUD:
    """Base CRUD class with common database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def commit(self) -> bool:
        """Commit transaction"""
        try:
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database commit error: {e}")
            return False
    
    def refresh(self, obj) -> None:
        """Refresh object state"""
        try:
            self.db.refresh(obj)
        except SQLAlchemyError as e:
            logger.error(f"Refresh error: {e}")


class MarketDataCRUD(CRUD):
    """CRUD operations for MarketData"""
    
    def create(self, market_data: models.MarketData) -> Optional[models.MarketData]:
        """Create new market data entry"""
        try:
            self.db.add(market_data)
            if self.commit():
                return market_data
            return None
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating market data: {e}")
            return None
    
    def bulk_create(self, market_data_list: List[models.MarketData]) -> bool:
        """Create multiple market data entries"""
        try:
            self.db.bulk_save_objects(market_data_list)
            self.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error bulk creating market data: {e}")
            return False
    
    def get_by_id(self, id: int) -> Optional[models.MarketData]:
        """Get market data by ID"""
        try:
            return self.db.query(models.MarketData).filter(models.MarketData.id == id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting market data by ID: {e}")
            return None
    
    def get_by_symbol_timeframe(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime = None, 
        end_date: datetime = None,
        limit: int = None
    ) -> List[models.MarketData]:
        """Get market data by symbol and timeframe with optional date range"""
        try:
            query = self.db.query(models.MarketData).filter(
                models.MarketData.symbol == symbol,
                models.MarketData.timeframe == timeframe
            )
            
            if start_date:
                query = query.filter(models.MarketData.timestamp >= start_date)
            if end_date:
                query = query.filter(models.MarketData.timestamp <= end_date)
            
            query = query.order_by(models.MarketData.timestamp.asc())
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting market data by symbol/timeframe: {e}")
            return []
    
    def get_latest(self, symbol: str, timeframe: str, limit: int = 100) -> List[models.MarketData]:
        """Get latest market data for symbol and timeframe"""
        try:
            return self.db.query(models.MarketData).filter(
                models.MarketData.symbol == symbol,
                models.MarketData.timeframe == timeframe
            ).order_by(models.MarketData.timestamp.desc()).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting latest market data: {e}")
            return []
    
    def get_ohlc(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get OHLC data for a specific period"""
        try:
            data = self.db.query(
                models.MarketData.timestamp,
                models.MarketData.open,
                models.MarketData.high,
                models.MarketData.low,
                models.MarketData.close,
                models.MarketData.volume
            ).filter(
                models.MarketData.symbol == symbol,
                models.MarketData.timeframe == timeframe,
                models.MarketData.timestamp >= start_date,
                models.MarketData.timestamp <= end_date
            ).order_by(models.MarketData.timestamp.asc()).all()
            
            return [
                {
                    'timestamp': row.timestamp,
                    'open': float(row.open),
                    'high': float(row.high),
                    'low': float(row.low),
                    'close': float(row.close),
                    'volume': float(row.volume)
                }
                for row in data
            ]
        except SQLAlchemyError as e:
            logger.error(f"Error getting OHLC data: {e}")
            return []
    
    def delete_by_symbol(self, symbol: str) -> int:
        """Delete all market data for a symbol"""
        try:
            count = self.db.query(models.MarketData).filter(
                models.MarketData.symbol == symbol
            ).delete()
            self.commit()
            return count
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting market data by symbol: {e}")
            return 0
    
    def get_missing_periods(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Identify missing data periods"""
        try:
            # This is a simplified implementation
            # In production, you'd need to implement based on your timeframe
            all_data = self.get_by_symbol_timeframe(symbol, timeframe, start_date, end_date)
            
            if not all_data:
                return [(start_date, end_date)]
            
            missing_periods = []
            all_data.sort(key=lambda x: x.timestamp)
            
            # Check gaps between consecutive data points
            for i in range(len(all_data) - 1):
                current = all_data[i].timestamp
                next_data = all_data[i + 1].timestamp
                
                # Assuming 1 minute interval for simplicity
                # Adjust based on your timeframe
                expected_next = current + timedelta(minutes=1)
                
                if next_data > expected_next + timedelta(minutes=5):  # 5 minute tolerance
                    missing_periods.append((expected_next, next_data - timedelta(seconds=1)))
            
            return missing_periods
        except Exception as e:
            logger.error(f"Error finding missing periods: {e}")
            return []


class TradeCRUD(CRUD):
    """CRUD operations for Trade"""
    
    def create(self, trade: models.Trade) -> Optional[models.Trade]:
        """Create new trade"""
        try:
            self.db.add(trade)
            if self.commit():
                return trade
            return None
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating trade: {e}")
            return None
    
    def get_by_trade_id(self, trade_id: str) -> Optional[models.Trade]:
        """Get trade by exchange trade ID"""
        try:
            return self.db.query(models.Trade).filter(
                models.Trade.trade_id == trade_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting trade by ID: {e}")
            return None
    
    def get_by_symbol(
        self, 
        symbol: str, 
        start_date: datetime = None, 
        end_date: datetime = None,
        side: str = None
    ) -> List[models.Trade]:
        """Get trades by symbol with optional filters"""
        try:
            query = self.db.query(models.Trade).filter(
                models.Trade.symbol == symbol
            )
            
            if start_date:
                query = query.filter(models.Trade.timestamp >= start_date)
            if end_date:
                query = query.filter(models.Trade.timestamp <= end_date)
            if side:
                query = query.filter(models.Trade.side == side)
            
            return query.order_by(models.Trade.timestamp.desc()).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting trades by symbol: {e}")
            return []
    
    def get_by_order_id(self, order_id: str) -> List[models.Trade]:
        """Get trades by exchange order ID"""
        try:
            return self.db.query(models.Trade).filter(
                models.Trade.exchange_order_id == order_id
            ).order_by(models.Trade.timestamp.asc()).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting trades by order ID: {e}")
            return []
    
    def get_today_trades(self, symbol: str = None) -> List[models.Trade]:
        """Get today's trades"""
        try:
            today = datetime.utcnow().date()
            start_of_day = datetime.combine(today, datetime.min.time())
            end_of_day = datetime.combine(today, datetime.max.time())
            
            query = self.db.query(models.Trade).filter(
                models.Trade.timestamp >= start_of_day,
                models.Trade.timestamp <= end_of_day
            )
            
            if symbol:
                query = query.filter(models.Trade.symbol == symbol)
            
            return query.order_by(models.Trade.timestamp.desc()).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting today's trades: {e}")
            return []
    
    def get_total_volume(self, symbol: str, start_date: datetime, end_date: datetime) -> float:
        """Get total trading volume for a period"""
        try:
            result = self.db.query(func.sum(models.Trade.quantity * models.Trade.price)).filter(
                models.Trade.symbol == symbol,
                models.Trade.timestamp >= start_date,
                models.Trade.timestamp <= end_date
            ).scalar()
            
            return float(result) if result else 0.0
        except SQLAlchemyError as e:
            logger.error(f"Error getting total volume: {e}")
            return 0.0
    
    def get_trade_stats(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get trade statistics for a period"""
        try:
            trades = self.get_by_symbol(symbol, start_date, end_date)
            
            if not trades:
                return {
                    'total_trades': 0,
                    'buy_trades': 0,
                    'sell_trades': 0,
                    'total_volume': 0.0,
                    'avg_trade_size': 0.0
                }
            
            buy_trades = [t for t in trades if t.side == 'buy']
            sell_trades = [t for t in trades if t.side == 'sell']
            
            total_volume = sum(float(t.quantity * t.price) for t in trades)
            avg_trade_size = total_volume / len(trades) if trades else 0
            
            return {
                'total_trades': len(trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'total_volume': total_volume,
                'avg_trade_size': avg_trade_size
            }
        except Exception as e:
            logger.error(f"Error getting trade stats: {e}")
            return {}


class OrderCRUD(CRUD):
    """CRUD operations for Order"""
    
    def create(self, order: models.Order) -> Optional[models.Order]:
        """Create new order"""
        try:
            self.db.add(order)
            if self.commit():
                return order
            return None
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating order: {e}")
            return None
    
    def update_status(
        self, 
        order_id: str, 
        status: str, 
        executed_quantity: float = None,
        updated_time: datetime = None
    ) -> bool:
        """Update order status"""
        try:
            order = self.db.query(models.Order).filter(
                models.Order.order_id == order_id
            ).first()
            
            if not order:
                return False
            
            order.status = status
            if executed_quantity is not None:
                order.executed_quantity = executed_quantity
            if updated_time:
                order.updated_time = updated_time
            else:
                order.updated_time = datetime.utcnow()
            
            return self.commit()
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating order status: {e}")
            return False
    
    def get_open_orders(self, symbol: str = None) -> List[models.Order]:
        """Get all open orders"""
        try:
            query = self.db.query(models.Order).filter(
                models.Order.status.in_(['pending', 'partially_filled'])
            )
            
            if symbol:
                query = query.filter(models.Order.symbol == symbol)
            
            return query.order_by(models.Order.created_time.desc()).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def get_by_session_id(self, session_id: int) -> List[models.Order]:
        """Get orders by trading session ID"""
        try:
            return self.db.query(models.Order).filter(
                models.Order.trading_session_id == session_id
            ).order_by(models.Order.created_time.desc()).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting orders by session: {e}")
            return []
    
    def get_order_summary(self, symbol: str) -> Dict[str, Any]:
        """Get order summary for symbol"""
        try:
            orders = self.db.query(models.Order).filter(
                models.Order.symbol == symbol
            ).all()
            
            summary = {
                'total_orders': len(orders),
                'open_orders': len([o for o in orders if o.status in ['pending', 'partially_filled']]),
                'filled_orders': len([o for o in orders if o.status == 'filled']),
                'cancelled_orders': len([o for o in orders if o.status == 'cancelled']),
                'buy_orders': len([o for o in orders if o.side == 'buy']),
                'sell_orders': len([o for o in orders if o.side == 'sell'])
            }
            
            return summary
        except SQLAlchemyError as e:
            logger.error(f"Error getting order summary: {e}")
            return {}


class TradingSessionCRUD(CRUD):
    """CRUD operations for TradingSession"""
    
    def create(self, session: models.TradingSession) -> Optional[models.TradingSession]:
        """Create new trading session"""
        try:
            self.db.add(session)
            if self.commit():
                return session
            return None
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating trading session: {e}")
            return None
    
    def get_active_sessions(self, strategy_name: str = None) -> List[models.TradingSession]:
        """Get active trading sessions"""
        try:
            query = self.db.query(models.TradingSession).filter(
                models.TradingSession.status == 'active'
            )
            
            if strategy_name:
                query = query.filter(models.TradingSession.strategy_name == strategy_name)
            
            return query.order_by(models.TradingSession.start_time.desc()).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    def update_capital(self, session_id: int, new_capital: float) -> bool:
        """Update session capital"""
        try:
            session = self.db.query(models.TradingSession).filter(
                models.TradingSession.id == session_id
            ).first()
            
            if not session:
                return False
            
            session.current_capital = new_capital
            return self.commit()
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating session capital: {e}")
            return False
    
    def end_session(self, session_id: int, end_time: datetime = None) -> bool:
        """End trading session"""
        try:
            session = self.db.query(models.TradingSession).filter(
                models.TradingSession.id == session_id
            ).first()
            
            if not session:
                return False
            
            session.status = 'stopped'
            session.end_time = end_time or datetime.utcnow()
            
            return self.commit()
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error ending session: {e}")
            return False
    
    def get_session_performance(self, session_id: int) -> Dict[str, Any]:
        """Get performance summary for session"""
        try:
            session = self.db.query(models.TradingSession).filter(
                models.TradingSession.id == session_id
            ).first()
            
            if not session:
                return {}
            
            orders = self.db.query(models.Order).filter(
                models.Order.trading_session_id == session_id
            ).all()
            
            trades = []
            for order in orders:
                trades.extend(order.trades)
            
            if not trades:
                return {
                    'session_id': session.session_id,
                    'strategy': session.strategy_name,
                    'initial_capital': float(session.initial_capital),
                    'current_capital': float(session.current_capital),
                    'total_return': 0.0,
                    'total_trades': 0
                }
            
            # Calculate simple P&L
            total_pnl = float(session.current_capital) - float(session.initial_capital)
            return_pct = (total_pnl / float(session.initial_capital)) * 100
            
            return {
                'session_id': session.session_id,
                'strategy': session.strategy_name,
                'initial_capital': float(session.initial_capital),
                'current_capital': float(session.current_capital),
                'total_pnl': total_pnl,
                'return_pct': return_pct,
                'total_trades': len(trades),
                'active_orders': len([o for o in orders if o.status in ['pending', 'partially_filled']])
            }
        except Exception as e:
            logger.error(f"Error getting session performance: {e}")
            return {}


class SignalCRUD(CRUD):
    """CRUD operations for Signal"""
    
    def create(self, signal: models.Signal) -> Optional[models.Signal]:
        """Create new signal"""
        try:
            self.db.add(signal)
            if self.commit():
                return signal
            return None
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating signal: {e}")
            return None
    
    def get_recent_signals(
        self, 
        symbol: str = None, 
        signal_type: str = None,
        limit: int = 50
    ) -> List[models.Signal]:
        """Get recent signals with optional filters"""
        try:
            query = self.db.query(models.Signal)
            
            if symbol:
                query = query.filter(models.Signal.symbol == symbol)
            if signal_type:
                query = query.filter(models.Signal.signal_type == signal_type)
            
            return query.order_by(models.Signal.timestamp.desc()).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    def get_signals_by_timeframe(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[models.Signal]:
        """Get signals by timeframe and date range"""
        try:
            return self.db.query(models.Signal).filter(
                models.Signal.symbol == symbol,
                models.Signal.timeframe == timeframe,
                models.Signal.timestamp >= start_date,
                models.Signal.timestamp <= end_date
            ).order_by(models.Signal.timestamp.asc()).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting signals by timeframe: {e}")
            return []
    
    def get_signal_accuracy(
        self, 
        source: str, 
        source_id: str, 
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """Calculate signal accuracy for a source"""
        try:
            start_date = datetime.utcnow() - timedelta(days=lookback_days)
            
            signals = self.db.query(models.Signal).filter(
                models.Signal.source == source,
                models.Signal.source_id == source_id,
                models.Signal.timestamp >= start_date
            ).all()
            
            if not signals:
                return {
                    'total_signals': 0,
                    'accuracy': 0.0,
                    'avg_confidence': 0.0
                }
            
            # This is a simplified accuracy calculation
            # In reality, you'd need to compare signals with actual price movements
            strong_signals = [s for s in signals if s.confidence and s.confidence > 0.7]
            
            accuracy = len(strong_signals) / len(signals) if signals else 0
            avg_confidence = sum(s.confidence or 0 for s in signals) / len(signals) if signals else 0
            
            return {
                'total_signals': len(signals),
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'strong_signals': len(strong_signals)
            }
        except Exception as e:
            logger.error(f"Error calculating signal accuracy: {e}")
            return {}


class ModelTrainingCRUD(CRUD):
    """CRUD operations for ModelTraining"""
    
    def create(self, training: models.ModelTraining) -> Optional[models.ModelTraining]:
        """Create new model training record"""
        try:
            self.db.add(training)
            if self.commit():
                return training
            return None
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating model training: {e}")
            return None
    
    def update_training_results(
        self, 
        training_id: str, 
        status: str = None,
        training_end: datetime = None,
        training_metrics: Dict = None,
        validation_metrics: Dict = None,
        test_metrics: Dict = None,
        model_path: str = None
    ) -> bool:
        """Update model training results"""
        try:
            training = self.db.query(models.ModelTraining).filter(
                models.ModelTraining.training_id == training_id
            ).first()
            
            if not training:
                return False
            
            if status:
                training.status = status
            if training_end:
                training.training_end = training_end
            if training_metrics:
                training.training_metrics = training_metrics
            if validation_metrics:
                training.validation_metrics = validation_metrics
            if test_metrics:
                training.test_metrics = test_metrics
            if model_path:
                training.model_path = model_path
            
            return self.commit()
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating training results: {e}")
            return False
    
    def get_latest_training(
        self, 
        model_name: str, 
        symbol: str, 
        status: str = 'completed'
    ) -> Optional[models.ModelTraining]:
        """Get latest completed training for model and symbol"""
        try:
            return self.db.query(models.ModelTraining).filter(
                models.ModelTraining.model_name == model_name,
                models.ModelTraining.symbol == symbol,
                models.ModelTraining.status == status
            ).order_by(models.ModelTraining.training_end.desc()).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting latest training: {e}")
            return None
    
    def get_training_history(
        self, 
        model_name: str = None, 
        symbol: str = None,
        limit: int = 20
    ) -> List[models.ModelTraining]:
        """Get training history with optional filters"""
        try:
            query = self.db.query(models.ModelTraining)
            
            if model_name:
                query = query.filter(models.ModelTraining.model_name == model_name)
            if symbol:
                query = query.filter(models.ModelTraining.symbol == symbol)
            
            return query.order_by(models.ModelTraining.training_start.desc()).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting training history: {e}")
            return []


class DatabaseManager:
    """Main database manager providing access to all CRUD operations"""
    
    def __init__(self, db: Session):
        self.db = db
        self.market_data = MarketDataCRUD(db)
        self.trades = TradeCRUD(db)
        self.orders = OrderCRUD(db)
        self.trading_sessions = TradingSessionCRUD(db)
        self.signals = SignalCRUD(db)
        self.model_trainings = ModelTrainingCRUD(db)
    
    def get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                'market_data': self.db.query(func.count(models.MarketData.id)).scalar() or 0,
                'trades': self.db.query(func.count(models.Trade.id)).scalar() or 0,
                'orders': self.db.query(func.count(models.Order.id)).scalar() or 0,
                'trading_sessions': self.db.query(func.count(models.TradingSession.id)).scalar() or 0,
                'signals': self.db.query(func.count(models.Signal.id)).scalar() or 0,
                'model_trainings': self.db.query(func.count(models.ModelTraining.id)).scalar() or 0,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return stats
        except SQLAlchemyError as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old market data
            market_data_deleted = self.db.query(models.MarketData).filter(
                models.MarketData.timestamp < cutoff_date
            ).delete(synchronize_session=False)
            
            # Delete old trades
            trades_deleted = self.db.query(models.Trade).filter(
                models.Trade.timestamp < cutoff_date
            ).delete(synchronize_session=False)
            
            # Delete old orders (keep filled orders for longer)
            orders_cutoff = datetime.utcnow() - timedelta(days=365)
            orders_deleted = self.db.query(models.Order).filter(
                models.Order.created_time < orders_cutoff
            ).delete(synchronize_session=False)
            
            self.commit()
            
            return {
                'market_data_deleted': market_data_deleted,
                'trades_deleted': trades_deleted,
                'orders_deleted': orders_deleted
            }
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error cleaning up old data: {e}")
            return {}


# Factory function to create database manager
def get_database_manager(db: Session) -> DatabaseManager:
    """Create and return a DatabaseManager instance"""
    return DatabaseManager(db)