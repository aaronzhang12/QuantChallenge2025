from backtester import TradingAlgorithm, Side, GameEvent, Trade, TICK, _clip_price
from typing import Optional
import collections

class MomentumMamba(TradingAlgorithm):
    """
    Trades on momentum swings, looking for market overreactions to scoring runs.
    """

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.score_diff_history = []
        self.momentum_threshold = 5  # Score run of 5-0 or more

    def on_game_event_update(self, event: GameEvent):
        if event.event_type == "SCORE":
            current_score_diff = event.home_score - event.away_score
            self.score_diff_history.append(current_score_diff)

            if len(self.score_diff_history) > 10:  # Look at the last 10 scoring events
                self.score_diff_history.pop(0)
                
                # Look for a significant run
                recent_run = self.score_diff_history[-1] - self.score_diff_history[0]

                if recent_run >= self.momentum_threshold:
                    # Home team has momentum, market might be overvaluing them. Sell.
                    if self.positions.get("BASKETBALL", 0) > -500:
                        self.place_market_order("BASKETBALL", Side.SELL, 100)
                elif recent_run <= -self.momentum_threshold:
                    # Away team has momentum, market might be undervaluing home team. Buy.
                    if self.positions.get("BASKETBALL", 0) < 500:
                        self.place_market_order("BASKETBALL", Side.BUY, 100)

class EventHorizon(TradingAlgorithm):
    """
    Analyzes individual game events, weighting them to find an edge.
    Focuses on high-impact events and player streaks.
    """

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.player_hot_streaks = {}  # Track recent points by player
        self.event_weights = {
            "THREE_POINT": 1.5,
            "DUNK": 1.2,
            "TWO_POINT": 1.0,
            "LAYUP": 1.0,
            "FREE_THROW": 0.5,
            "STEAL": 0.8,
            "BLOCK": 0.8,
            "TURNOVER": -0.8
        }

    def on_game_event_update(self, event: GameEvent):
        impact = 0
        if event.player_name:
            if event.event_type == "SCORE":
                points = 0
                if event.shot_type == "THREE_POINT":
                    points = 3
                elif event.shot_type in ["DUNK", "TWO_POINT", "LAYUP"]:
                    points = 2
                elif event.shot_type == "FREE_THROW":
                    points = 1
                
                self.player_hot_streaks.setdefault(event.player_name, []).append(points)
                if len(self.player_hot_streaks[event.player_name]) > 3:
                    self.player_hot_streaks[event.player_name].pop(0)

                # If a player is on a hot streak, amplify the impact
                if sum(self.player_hot_streaks.get(event.player_name, [])) >= 5: # 5+ points in last 3 scoring plays
                    impact += self.event_weights.get(event.shot_type, 0) * 1.5
                else:
                    impact += self.event_weights.get(event.shot_type, 0)

            elif event.event_type in self.event_weights:
                 impact += self.event_weights[event.event_type]

            if event.home_away == "away":
                impact *= -1

        if impact > 1.0: # Significant positive event for home team
            if self.positions.get("BASKETBALL", 0) < 500:
                self.place_market_order("BASKETBALL", Side.BUY, 50 * impact)
        elif impact < -1.0: # Significant negative event for home team
            if self.positions.get("BASKETBALL", 0) > -500:
                self.place_market_order("BASKETBALL", Side.SELL, 50 * abs(impact))

class ArbitrageAce(TradingAlgorithm):
    """
    Identifies and trades on discrepancies between the fair value and the market price.
    """

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.fair_value = 50.0
        self.last_market_price = 50.0
        self.mispricing_threshold = 1.5 

    def on_trade_update(self, trade: Trade):
        if trade.ticker == "BASKETBALL":
            self.last_market_price = trade.price

    def on_game_event_update(self, event: GameEvent):
        score_diff = event.home_score - event.away_score
        time_seconds = event.time_seconds if event.time_seconds is not None else 0
        time_factor = max(0.1, time_seconds / 2880.0)
        score_adjustment = score_diff * (2.0 - time_factor) * 0.5
        self.fair_value = max(5.0, min(95.0, 50.0 + score_adjustment))

        mispricing = self.fair_value - self.last_market_price
        
        position_limit = 500
        current_position = self.positions.get("BASKETBALL", 0)

        if mispricing > self.mispricing_threshold and current_position < position_limit:
            order_size = min(150, int(50 * mispricing))
            if order_size > 0: self.place_market_order("BASKETBALL", Side.BUY, order_size)

        elif mispricing < -self.mispricing_threshold and current_position > -position_limit:
            order_size = min(150, int(50 * abs(mispricing)))
            if order_size > 0: self.place_market_order("BASKETBALL", Side.SELL, order_size)

class VolatilityVulture(TradingAlgorithm):
    """
    Adapts its strategy based on market volatility.
    """

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.fair_value = 50.0
        self.base_spread = 1.5
        self.order_size = 75
        self.bid_order_id = None
        self.ask_order_id = None
        self.price_history = collections.deque(maxlen=20)

    def on_trade_update(self, trade: Trade):
        if trade.ticker == "BASKETBALL":
            self.price_history.append(trade.price)

    def on_game_event_update(self, event: GameEvent):
        score_diff = event.home_score - event.away_score
        time_seconds = event.time_seconds if event.time_seconds is not None else 0
        time_factor = max(0.1, time_seconds / 2880.0)
        self.fair_value = max(5.0, min(95.0, 50.0 + score_diff * (2.0 - time_factor) * 0.5))
        
        self._update_quotes()

    def _calculate_volatility(self) -> float:
        if len(self.price_history) < 5:
            return 0.0
        
        prices = list(self.price_history)
        mean_price = sum(prices) / len(prices)
        variance = sum([(p - mean_price) ** 2 for p in prices]) / len(prices)
        std_dev = variance ** 0.5
        return std_dev

    def _update_quotes(self):
        volatility = self._calculate_volatility()
        volatility_adjustment = volatility * 2.5
        adjusted_spread = self.base_spread + volatility_adjustment

        bid_price = self.fair_value - adjusted_spread / 2
        ask_price = self.fair_value + adjusted_spread / 2

        bid_price = max(1.0, min(bid_price, 99.0))
        ask_price = max(1.0, min(ask_price, 99.0))

        if self.bid_order_id:
            self.cancel_order("BASKETBALL", self.bid_order_id)
        if self.ask_order_id:
            self.cancel_order("BASKETBALL", self.ask_order_id)

        position_limit = 750
        current_position = self.positions.get("BASKETBALL", 0)

        if current_position < position_limit:
            self.bid_order_id = self.place_limit_order("BASKETBALL", Side.BUY, self.order_size, bid_price)

        if current_position > -position_limit:
            self.ask_order_id = self.place_limit_order("BASKETBALL", Side.SELL, self.order_size, ask_price)