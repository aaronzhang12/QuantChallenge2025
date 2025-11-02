"""
Quant Challenge 2025

Algorithmic strategy template
"""

from backtester import TradingAlgorithm, Side, GameEvent, Trade
from typing import Optional, List, Tuple
import math


class Strategy(TradingAlgorithm):
    """
    A strategy that calculates a fair value for the home team's win probability
    based on score and time remaining, then trades when the market price deviates.
    """

    TRADING_THRESHOLD = 2.0
    MAX_POSITION = 50
    STARTING_CAPITAL = 100000.0

    def __init__(self, trader_id: str, capital: float = 100000.0) -> None:
        super().__init__(trader_id, capital)
        self.reset_state()

    def reset_state(self) -> None:
        print("--- RESETTING STRATEGY STATE FOR NEW GAME ---")
        self.best_bid_price: Optional[float] = None
        self.best_ask_price: Optional[float] = None
        self.fair_value = 50.0

    def _calculate_fair_value(self, home_score: int, away_score: int, time_seconds: float) -> float:
        if time_seconds is None or time_seconds <= 0:
            return 100.0 if home_score > away_score else 0.0

        score_diff = home_score - away_score
        lead_factor = score_diff / math.sqrt(time_seconds + 1) * 2.5
        probability = 1 / (1 + math.exp(-lead_factor))

        return probability * 100

    def _execute_trade_logic(self) -> None:
        if self.best_ask_price is None or self.best_bid_price is None:
            return

        current_position = self.positions["BASKETBALL"]

        desired_position = 0.0
        if self.fair_value > self.best_ask_price + self.TRADING_THRESHOLD:
            desired_position = self.MAX_POSITION
        elif self.fair_value < self.best_bid_price - self.TRADING_THRESHOLD:
            desired_position = -self.MAX_POSITION
        else:
            desired_position = 0.0

        trade_quantity = desired_position - current_position

        if abs(trade_quantity) > 0.1:
            side = Side.BUY if trade_quantity > 0 else Side.SELL
            self.place_market_order("BASKETBALL", side, abs(trade_quantity))

    def on_orderbook_update(self, ticker: str, price: float, quantity: int):
        if quantity > 0:
            if self.best_bid_price is None or price > self.best_bid_price:
                self.best_bid_price = price
            if self.best_ask_price is None or price < self.best_ask_price:
                self.best_ask_price = price

    def on_trade_update(self, trade: Trade):
        if trade.ticker == "BASKETBALL":
            self.best_bid_price = trade.price - 1.0
            self.best_ask_price = trade.price + 1.0

    def on_account_update(self, order_id: str, quantity: int, price: float, capital_remaining: float):
        current_position = self.positions["BASKETBALL"]
        print(f"ACCOUNT UPDATE: Position is now {current_position}, Capital is {capital_remaining:.2f}")

    def on_game_event_update(self, event: GameEvent):
        if event.event_type == "END_GAME":
            print(f"FINAL SCORE: {event.home_score} - {event.away_score}")
            self.reset_state()
            return

        self.fair_value = self._calculate_fair_value(event.home_score, event.away_score, event.time_seconds)
        print(f"GAME UPDATE: Time: {event.time_seconds:.1f}s, Score: {event.home_score}-{event.away_score}, New Fair Value: ${self.fair_value:.2f}")
        self._execute_trade_logic()


class QuantitativeWinningStrategy(TradingAlgorithm):

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.fair_value = 50.0
        self.best_bid = None
        self.best_ask = None
        self.last_market_price = 50.0
        self.max_position = 10
        self.trade_size = 5
        self.min_edge_required = 5.0
        self.last_trade_time = 0
        self.trades_made = 0
        self.max_trades_per_game = 10

    def _calculate_simple_fair_value(self, home_score: int, away_score: int, time_seconds: float) -> float:
        if time_seconds <= 0:
            return 100.0 if home_score > away_score else 0.0

        score_diff = home_score - away_score
        time_factor = max(0.01, time_seconds / 2880)
        multiplier = 0.3 + (1.0 - time_factor) * 0.5
        probability = 1 / (1 + math.exp(-score_diff * multiplier))

        return max(5.0, min(95.0, probability * 100))

    def _should_trade(self, edge: float, time_seconds: float) -> bool:
        if abs(time_seconds - self.last_trade_time) < 30.0:
            return False
        if self.trades_made >= self.max_trades_per_game:
            return False
        if abs(edge) < self.min_edge_required:
            return False
        if time_seconds > 2760 or time_seconds < 120:
            return False
        return True

    def _execute_conservative_trade(self, edge: float, time_seconds: float):
        side = Side.BUY if edge > 0 else Side.SELL
        if side == Side.BUY:
            limit_price = self.last_market_price - 1.0
        else:
            limit_price = self.last_market_price + 1.0

        limit_price = max(1.0, min(99.0, limit_price))
        order_id = self.place_limit_order("BASKETBALL", side, self.trade_size, limit_price, ioc=True)
        self.last_trade_time = time_seconds
        self.trades_made += 1
        print(f"CONSERVATIVE: {side.name} {self.trade_size} @ {limit_price:.2f} (edge: {edge:.1f}, trade #{self.trades_made})")

    def _close_position_if_needed(self, time_seconds: float):
        current_position = self.positions["BASKETBALL"]
        if abs(current_position) > 0 and time_seconds < 180:
            close_side = Side.SELL if current_position > 0 else Side.BUY
            close_size = min(self.trade_size, abs(current_position))

            if close_side == Side.SELL:
                close_price = self.last_market_price - 2.0
            else:
                close_price = self.last_market_price + 2.0

            close_price = max(1.0, min(99.0, close_price))
            self.place_limit_order("BASKETBALL", close_side, close_size, close_price, ioc=True)
            print(f"END GAME CLOSE: {close_side.name} {close_size} @ {close_price:.2f}")

    def on_game_event_update(self, event: GameEvent):
        if event.event_type == "END_GAME":
            print(f"GAME END: Home {event.home_score} - Away {event.away_score}")
            self._reset_game_state()
            return

        self.fair_value = self._calculate_simple_fair_value(event.home_score, event.away_score, event.time_seconds)
        raw_edge = self.fair_value - self.last_market_price
        edge = raw_edge - 1.0
        current_position = self.positions["BASKETBALL"]

        if self._should_trade(edge, event.time_seconds):
            if edge > 0 and current_position < self.max_position:
                self._execute_conservative_trade(edge, event.time_seconds)
            elif edge < 0 and current_position > -self.max_position:
                self._execute_conservative_trade(edge, event.time_seconds)

        self._close_position_if_needed(event.time_seconds)

        if event.event_type == "SCORE":
            print(f"QUANT SCORE: {event.home_score}-{event.away_score} | Fair=${self.fair_value:.1f} | Edge={edge:.1f} | Pos={current_position}")

    def on_trade_update(self, trade: Trade):
        if trade.ticker == "BASKETBALL":
            self.last_market_price = trade.price

    def on_account_update(self, order_id: str, quantity: int, price: float, capital_remaining: float):
        current_position = self.positions["BASKETBALL"]
        side_str = "BUY" if quantity > 0 else "SELL"
        print(f"QUANT FILL: {side_str} {abs(quantity)} @ ${price:.2f} | Pos={current_position:.0f} | Capital=${capital_remaining:.0f}")

    def on_orderbook_update(self, ticker: str, price: float, quantity: int):
        if ticker == "BASKETBALL":
            if not hasattr(self, 'best_bid') or price > (self.best_bid or 0):
                self.best_bid = price
            if not hasattr(self, 'best_ask') or price < (self.best_ask or 100):
                self.best_ask = price

    def _reset_game_state(self):
        self.fair_value = 50.0
        self.last_trade_time = 0
        self.trades_made = 0
        self.last_market_price = 50.0