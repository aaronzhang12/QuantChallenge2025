"""
ADVANCED ADVERSARIAL BOTS - Tournament Competition Enhancement

These sophisticated bots employ advanced trading strategies to create
a highly competitive and realistic tournament environment.
"""

import random
import math
from collections import deque
from typing import Dict, List, Optional, Tuple
from backtester import TradingAlgorithm, Side, GameEvent, Trade


class StatisticalArbitrageBot(TradingAlgorithm):
    """
    Pairs trading bot that looks for statistical relationships between
    score differentials and market prices, trading mean reversion.
    """
    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.score_price_history = []  # (score_diff, price) pairs
        self.lookback = 50
        self.z_score_threshold = 1.8
        self.max_position = 300
        self.base_size = 40

    def on_game_event_update(self, event: GameEvent):
        score_diff = event.home_score - event.away_score
        current_price = self._estimate_market_price()

        self.score_price_history.append((score_diff, current_price))
        if len(self.score_price_history) > self.lookback:
            self.score_price_history.pop(0)

        if len(self.score_price_history) < 20:
            return

        # Calculate expected price from historical relationship
        expected_price = self._calculate_expected_price(score_diff)

        if expected_price is None:
            return

        deviation = current_price - expected_price
        z_score = self._calculate_z_score(deviation)

        if abs(z_score) > self.z_score_threshold:
            side = Side.SELL if z_score > 0 else Side.BUY
            size = min(self.base_size * int(abs(z_score)),
                      abs(self.max_position - abs(self.positions["BASKETBALL"])))

            if size > 10:
                price = current_price + (0.3 if side == Side.BUY else -0.3)
                price = max(1.0, min(99.0, price))
                self.place_limit_order("BASKETBALL", side, size, price, ioc=True)

    def _estimate_market_price(self) -> float:
        # Simple estimation - would use actual market data in real implementation
        return 50.0 + random.gauss(0, 5)

    def _calculate_expected_price(self, score_diff: int) -> Optional[float]:
        if len(self.score_price_history) < 10:
            return None

        # Simple linear regression approximation
        similar_diffs = [price for diff, price in self.score_price_history
                        if abs(diff - score_diff) <= 2]
        return sum(similar_diffs) / len(similar_diffs) if similar_diffs else None

    def _calculate_z_score(self, deviation: float) -> float:
        deviations = [cp - ep for sd, cp in self.score_price_history
                     for ep in [self._calculate_expected_price(sd)] if ep]
        if len(deviations) < 5:
            return 0.0

        mean_dev = sum(deviations) / len(deviations)
        std_dev = (sum((d - mean_dev) ** 2 for d in deviations) / len(deviations)) ** 0.5
        return (deviation - mean_dev) / (std_dev + 0.01)


class HighFrequencyScalper(TradingAlgorithm):
    """
    Ultra-fast scalping bot that captures tiny spreads with high frequency,
    simulating HFT-like behavior in the basketball market.
    """
    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.tick_count = 0
        self.scalp_size = 25
        self.max_position = 150
        self.target_spread = 0.4
        self.hold_time_limit = 8  # Events to hold position
        self.position_entry_time = {}

    def on_game_event_update(self, event: GameEvent):
        self.tick_count += 1

        # Close aged positions
        self._close_aged_positions()

        # Scalp on every 2nd event for high frequency
        if self.tick_count % 2 == 0:
            self._attempt_scalp(event)

    def _attempt_scalp(self, event: GameEvent):
        current_pos = self.positions["BASKETBALL"]

        # Don't exceed position limits
        if abs(current_pos) >= self.max_position:
            return

        # Estimate current mid price
        mid_price = 50.0 + (event.home_score - event.away_score) * 1.5
        mid_price = max(5.0, min(95.0, mid_price))

        # Random scalp direction with slight bias based on momentum
        momentum_bias = 1 if event.event_type == "SCORE" and event.home_away == "home" else -1
        direction = random.choice([1, -1, momentum_bias])

        if direction > 0 and current_pos < self.max_position:
            # Scalp long
            entry_price = mid_price - 0.2  # Aggressive bid
            self.place_limit_order("BASKETBALL", Side.BUY, self.scalp_size, entry_price, ioc=True)
            self.position_entry_time[self.tick_count] = ("LONG", self.scalp_size)
        elif direction < 0 and current_pos > -self.max_position:
            # Scalp short
            entry_price = mid_price + 0.2  # Aggressive offer
            self.place_limit_order("BASKETBALL", Side.SELL, self.scalp_size, entry_price, ioc=True)
            self.position_entry_time[self.tick_count] = ("SHORT", self.scalp_size)

    def _close_aged_positions(self):
        positions_to_close = []
        for entry_tick, (direction, size) in self.position_entry_time.items():
            if self.tick_count - entry_tick >= self.hold_time_limit:
                positions_to_close.append(entry_tick)

                # Close with market order for fast exit
                close_side = Side.SELL if direction == "LONG" else Side.BUY
                self.place_market_order("BASKETBALL", close_side, size)

        for tick in positions_to_close:
            del self.position_entry_time[tick]


class GameFlowSpecialist(TradingAlgorithm):
    """
    Sophisticated bot that understands basketball game flow patterns,
    trading based on quarter timing, momentum shifts, and situational contexts.
    """
    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.max_position = 400
        self.base_size = 35
        self.quarter_patterns = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.momentum_events = deque(maxlen=10)
        self.last_significant_event = None

    def on_game_event_update(self, event: GameEvent):
        quarter = self._get_quarter(event.time_seconds)
        self._track_momentum(event)

        # Quarter-based strategies
        quarter_bias = self._get_quarter_bias(quarter)

        # Momentum analysis
        momentum_signal = self._analyze_momentum()

        # Situational context
        situation_signal = self._analyze_situation(event, quarter)

        # Combine signals
        combined_signal = quarter_bias + momentum_signal + situation_signal

        if abs(combined_signal) > 2.0:  # Strong signal threshold
            side = Side.BUY if combined_signal > 0 else Side.SELL

            # Dynamic sizing based on signal strength
            size = min(int(self.base_size * (abs(combined_signal) / 2.0)),
                      self._get_max_trade_size(side))

            if size > 15:
                price = self._get_aggressive_price(side, event)
                self.place_limit_order("BASKETBALL", side, size, price, ioc=True)

    def _get_quarter(self, time_seconds: Optional[float]) -> int:
        if not time_seconds:
            return 1
        # Basketball game: 4 quarters of 12 minutes = 720 seconds each
        return min(4, int(time_seconds // 720) + 1)

    def _track_momentum(self, event: GameEvent):
        momentum_value = 0
        if event.event_type == "SCORE":
            points = 3 if event.shot_type == "THREE_POINT" else 2
            momentum_value = points if event.home_away == "home" else -points
        elif event.event_type in ["STEAL", "BLOCK"]:
            momentum_value = 1 if event.home_away == "home" else -1
        elif event.event_type == "TURNOVER":
            momentum_value = -1 if event.home_away == "home" else 1

        if momentum_value != 0:
            self.momentum_events.append((event.time_seconds, momentum_value))

    def _get_quarter_bias(self, quarter: int) -> float:
        # 4th quarter is most important, 3rd quarter often has momentum swings
        quarter_weights = {1: 0.5, 2: 0.3, 3: 1.2, 4: 1.8}
        return self.quarter_patterns.get(quarter, 0.0) * quarter_weights[quarter]

    def _analyze_momentum(self) -> float:
        if len(self.momentum_events) < 3:
            return 0.0

        recent_momentum = sum(value for _, value in list(self.momentum_events)[-5:])
        momentum_strength = abs(recent_momentum)

        # Fade strong momentum (contrarian)
        if momentum_strength > 8:
            return -recent_momentum * 0.3
        # Follow moderate momentum
        elif momentum_strength > 3:
            return recent_momentum * 0.2

        return 0.0

    def _analyze_situation(self, event: GameEvent, quarter: int) -> float:
        situation_signal = 0.0

        # End of quarter desperation
        if quarter == 4 and event.time_seconds and event.time_seconds < 120:  # Last 2 minutes
            score_diff = event.home_score - event.away_score
            if abs(score_diff) > 10:  # Blowout - fade
                situation_signal = -score_diff * 0.1
            elif abs(score_diff) < 3:  # Close game - volatility
                situation_signal = random.choice([-1.5, 1.5])

        # Three-point barrage situations
        if event.event_type == "SCORE" and event.shot_type == "THREE_POINT":
            # Teams often go on three-point runs
            situation_signal = 1.0 if event.home_away == "home" else -1.0

        return situation_signal

    def _get_max_trade_size(self, side: Side) -> int:
        current_pos = self.positions["BASKETBALL"]
        if side == Side.BUY:
            return max(0, self.max_position - current_pos)
        else:
            return max(0, self.max_position + current_pos)

    def _get_aggressive_price(self, side: Side, event: GameEvent) -> float:
        base_price = 50.0 + (event.home_score - event.away_score) * 2.0

        if side == Side.BUY:
            price = base_price + 0.6  # Aggressive bid
        else:
            price = base_price - 0.6  # Aggressive offer

        return max(1.0, min(99.0, price))


class PsychologicalManipulator(TradingAlgorithm):
    """
    Advanced bot that attempts to manipulate market psychology through
    strategic order placement and creates false signals for other algorithms.
    """
    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.max_position = 250
        self.manipulation_cycle = 0
        self.fake_signal_strength = 0
        self.target_bots = ["Simple", "Advanced", "Momentum", "Noise"]  # Bot types to manipulate

    def on_game_event_update(self, event: GameEvent):
        self.manipulation_cycle += 1

        # Alternate between different manipulation strategies
        strategy = self.manipulation_cycle % 4

        if strategy == 0:
            self._create_false_momentum(event)
        elif strategy == 1:
            self._spoof_orderbook(event)
        elif strategy == 2:
            self._pump_and_dump(event)
        else:
            self._contrarian_trap(event)

    def _create_false_momentum(self, event: GameEvent):
        """Create fake momentum to trick momentum-following bots."""
        # Place series of small orders in same direction to create momentum illusion
        direction = random.choice([Side.BUY, Side.SELL])

        for i in range(3):
            size = random.randint(10, 25)
            price_offset = 0.2 + i * 0.1

            if direction == Side.BUY:
                price = 50.0 + price_offset
            else:
                price = 50.0 - price_offset

            price = max(1.0, min(99.0, price))
            self.place_limit_order("BASKETBALL", direction, size, price, ioc=True)

    def _spoof_orderbook(self, event: GameEvent):
        """Place and quickly cancel large orders to create false liquidity signals."""
        # Large fake bid/offer to influence other algos
        fake_size = random.randint(100, 200)
        current_price = 50.0 + (event.home_score - event.away_score) * 1.5

        # Place fake order far from market
        if random.random() > 0.5:
            fake_price = current_price - 2.0  # Fake support
            order_id = self.place_limit_order("BASKETBALL", Side.BUY, fake_size, fake_price)
        else:
            fake_price = current_price + 2.0  # Fake resistance
            order_id = self.place_limit_order("BASKETBALL", Side.SELL, fake_size, fake_price)

        # Cancel after brief period (would need timer in real implementation)
        if order_id and random.random() < 0.3:  # 30% chance to cancel immediately
            self.cancel_order("BASKETBALL", order_id)

    def _pump_and_dump(self, event: GameEvent):
        """Artificially pump price then dump position."""
        current_pos = self.positions["BASKETBALL"]

        if current_pos < 50:  # Accumulation phase
            # Buy aggressively to pump price
            self.place_market_order("BASKETBALL", Side.BUY, 30)
        elif current_pos > 100:  # Distribution phase
            # Dump position
            self.place_market_order("BASKETBALL", Side.SELL, 60)

    def _contrarian_trap(self, event: GameEvent):
        """Set up contrarian traps for mean-reversion algorithms."""
        # Create artificial extreme to trap contrarian bots
        score_diff = event.home_score - event.away_score

        if abs(score_diff) > 8:  # Game situation suggests extreme
            # Push price even more extreme to trigger contrarians
            if score_diff > 0:  # Home team winning
                # Push price higher to trap shorts
                self.place_market_order("BASKETBALL", Side.BUY, 40)
            else:  # Away team winning
                # Push price lower to trap longs
                self.place_market_order("BASKETBALL", Side.SELL, 40)


class RiskParityBot(TradingAlgorithm):
    """
    Sophisticated risk management bot that sizes positions based on
    volatility and correlation patterns, maintaining constant risk exposure.
    """
    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.target_vol = 0.15  # Target portfolio volatility
        self.price_returns = deque(maxlen=30)
        self.current_vol = 0.02
        self.max_position = 300
        self.rebalance_counter = 0

    def on_game_event_update(self, event: GameEvent):
        self.rebalance_counter += 1

        # Update volatility estimates
        self._update_volatility_estimate(event)

        # Rebalance every few events
        if self.rebalance_counter % 5 == 0:
            self._rebalance_portfolio(event)

    def _update_volatility_estimate(self, event: GameEvent):
        current_price = 50.0 + (event.home_score - event.away_score) * 2.0

        if len(self.price_returns) > 0:
            last_price = self.price_returns[-1] if self.price_returns else 50.0
            return_pct = (current_price - last_price) / last_price
            self.price_returns.append(return_pct)

            # Calculate rolling volatility
            if len(self.price_returns) > 5:
                mean_return = sum(self.price_returns) / len(self.price_returns)
                variance = sum((r - mean_return) ** 2 for r in self.price_returns) / len(self.price_returns)
                self.current_vol = math.sqrt(variance)

        self.price_returns.append(current_price)

    def _rebalance_portfolio(self, event: GameEvent):
        if self.current_vol < 0.001:  # Avoid division by zero
            return

        # Calculate optimal position size based on risk parity
        vol_target_multiplier = self.target_vol / self.current_vol
        base_position = 100  # Base position size

        optimal_position = int(base_position * vol_target_multiplier)
        optimal_position = max(-self.max_position, min(self.max_position, optimal_position))

        current_position = self.positions["BASKETBALL"]
        position_diff = optimal_position - current_position

        # Execute rebalancing trade if significant difference
        if abs(position_diff) > 15:
            side = Side.BUY if position_diff > 0 else Side.SELL
            trade_size = min(abs(position_diff), 50)  # Trade in chunks

            # Use mid-price for rebalancing
            mid_price = 50.0 + (event.home_score - event.away_score) * 1.8
            self.place_limit_order("BASKETBALL", side, trade_size, mid_price, ioc=True)
