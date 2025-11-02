import numpy as np
from enum import Enum
from typing import Optional

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None: return
def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int: return 0
def cancel_order(ticker: Ticker, order_id: int) -> bool: return True

class Strategy:

    def reset_state(self) -> None:
        self.POSITION_SIZE = 1.0
        self.ENTRY_THRESHOLD = 2.0
        self.TAKE_PROFIT = 1.5
        self.STOP_LOSS = 2.0
        self.GAME_TOTAL_SECONDS = 2880.0
        self.PROFIT_TARGET = 100.0
        self.current_position = 0.0
        self.entry_price = 0.0
        self.exit_order_id = None
        self.realized_pnl = 0.0
        self.trading_enabled = True
        self.fair_value = 50.0
        self.best_bid = 0.0
        self.best_ask = 100.0
        self.time_seconds = self.GAME_TOTAL_SECONDS
        self.home_score = 0
        self.away_score = 0

    def __init__(self) -> None:
        self.reset_state()

    def _calculate_fair_value(self) -> float:
        if self.time_seconds is None or self.time_seconds <= 0:
            return 100.0 if self.home_score > self.away_score else 0.0
        score_diff = self.home_score - self.away_score
        k = 0.05
        time_decay_factor = self.time_seconds / self.GAME_TOTAL_SECONDS
        win_prob = 1 / (1 + np.exp(-k * score_diff / (time_decay_factor**0.5 + 0.01)))
        return win_prob * 100

    def _check_trading_logic(self) -> None:
        if not self.trading_enabled:
            return

        if self.current_position > 0:
            if self.best_bid < self.entry_price - self.STOP_LOSS:
                if self.exit_order_id is not None: cancel_order(Ticker.TEAM_A, self.exit_order_id)
                place_market_order(Side.SELL, Ticker.TEAM_A, self.current_position)
                return
        elif self.current_position < 0:
            if self.best_ask > self.entry_price + self.STOP_LOSS:
                if self.exit_order_id is not None: cancel_order(Ticker.TEAM_A, self.exit_order_id)
                place_market_order(Side.BUY, Ticker.TEAM_A, abs(self.current_position))
                return

        if self.current_position == 0:
            if self.fair_value > self.best_ask + self.ENTRY_THRESHOLD:
                self.entry_price = self.best_ask
                place_limit_order(Side.BUY, Ticker.TEAM_A, self.POSITION_SIZE, self.entry_price, ioc=True)
            elif self.fair_value < self.best_bid - self.ENTRY_THRESHOLD:
                self.entry_price = self.best_bid
                place_limit_order(Side.SELL, Ticker.TEAM_A, self.POSITION_SIZE, self.entry_price, ioc=True)

    def on_account_update(
        self, ticker: Ticker, side: Side, price: float, quantity: float, capital_remaining: float,
    ) -> None:
        print(f"$$$ FILL: {side} {quantity} shares @ {price}")

        is_closing_trade = (side == Side.SELL and self.current_position > 0) or \
                           (side == Side.BUY and self.current_position < 0)

        is_opening_trade = self.current_position == 0

        if is_closing_trade:
            pnl_for_this_trade = 0
            if side == Side.SELL:
                pnl_for_this_trade = (price - self.entry_price) * quantity
            else:
                pnl_for_this_trade = (self.entry_price - price) * quantity

            self.realized_pnl += pnl_for_this_trade
            print(f"### Position Closed. PnL for trade: ${pnl_for_this_trade:.2f}. Total PnL: ${self.realized_pnl:.2f} ###")

            self.current_position = 0.0
            self.exit_order_id = None

            if self.realized_pnl >= self.PROFIT_TARGET:
                print(f"!!! PROFIT TARGET of ${self.PROFIT_TARGET} REACHED! CEASING TRADING. !!!")
                self.trading_enabled = False

        elif is_opening_trade:
            if side == Side.BUY:
                self.current_position = quantity
                take_profit_price = price + self.TAKE_PROFIT
                self.exit_order_id = place_limit_order(Side.SELL, Ticker.TEAM_A, quantity, take_profit_price)
            else:
                self.current_position = -quantity
                take_profit_price = price - self.TAKE_PROFIT
                self.exit_order_id = place_limit_order(Side.BUY, Ticker.TEAM_A, quantity, take_profit_price)
            self.entry_price = price

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        if bids: self.best_bid = bids[0][0]
        if asks: self.best_ask = asks[0][0]
        self._check_trading_logic()

    def on_game_event_update(self, event_type: str, home_away: str, home_score: int, away_score: int, player_name: Optional[str], substituted_player_name: Optional[str], shot_type: Optional[str], assist_player: Optional[str], rebound_type: Optional[str], coordinate_x: Optional[float], coordinate_y: Optional[float], time_seconds: Optional[float]) -> None:
        if event_type == "END_GAME":
            self.reset_state()
            return
        self.home_score, self.away_score = home_score, away_score
        if time_seconds is not None: self.time_seconds = time_seconds
        self.fair_value = self._calculate_fair_value()
        self._check_trading_logic()
        
    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None: pass
    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None: pass