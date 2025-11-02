import random
import math
from typing import Dict, Optional
from backtester import TradingAlgorithm, Side, GameEvent, Trade, TICK, _clip_price


# ---------- Helpers ----------
def _safe_price(p: float) -> float:
    return _clip_price(p)

# ---------- Updated SimpleLiquidityBot ----------
class SimpleLiquidityBot(TradingAlgorithm):
    """Basic market maker that provides liquidity around a fair price estimate."""

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.fair_price = 50.0
        self.spread = 2.0
        self.order_size = 100
        self.max_position = 1000
        self.bid_order_id = None
        self.ask_order_id = None
        self.best_bid = None
        self.best_ask = None

    def on_orderbook_update(self, ticker: str, side: Side, price: float, quantity: int):
        if ticker != "BASKETBALL":
            return
        if side == Side.BUY:
            self.best_bid = price if (self.best_bid is None or price > self.best_bid) else self.best_bid
        else:
            self.best_ask = price if (self.best_ask is None or price < self.best_ask) else self.best_ask

    def on_game_event_update(self, event: GameEvent):
        self._update_fair_price(event)
        self._update_quotes("BASKETBALL", event.time_seconds)

    def _update_fair_price(self, event: GameEvent):
        score_diff = event.home_score - event.away_score
        time_factor = max(0.1, event.time_seconds / 2880.0)
        score_adjustment = score_diff * (2.0 - time_factor) * 0.5
        self.fair_price = max(5.0, min(95.0, 50.0 + score_adjustment))

    def _update_quotes(self, ticker: str, time_seconds: float):
        current_position = self.positions[ticker]
        # widen a touch late-game
        time_widen = 0.0 if time_seconds > 120 else 1.0
        adjusted_spread = self.spread + abs(current_position) / self.max_position * 2.0 + time_widen

        bid_price = _safe_price(self.fair_price - adjusted_spread / 2)
        ask_price = _safe_price(self.fair_price + adjusted_spread / 2)

        if self.bid_order_id:
            self.cancel_order(ticker, self.bid_order_id)
        if self.ask_order_id:
            self.cancel_order(ticker, self.ask_order_id)

        if current_position < self.max_position and self.capital_remaining > ask_price * self.order_size:
            self.bid_order_id = self.place_limit_order(ticker, Side.BUY, self.order_size, bid_price)

        if current_position > -self.max_position:
            self.ask_order_id = self.place_limit_order(ticker, Side.SELL, self.order_size, ask_price)

# ---------- Updated AdvancedLiquidityBot ----------
class AdvancedLiquidityBot(TradingAlgorithm):
    """More sophisticated market maker with momentum and volatility considerations."""

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.fair_price = 50.0
        self.momentum = 0.0
        self.volatility = 2.0
        self.recent_trades = []
        self.order_size = 150
        self.max_position = 800
        self.quote_orders = []
        self.best_bid = None
        self.best_ask = None

    def on_orderbook_update(self, ticker: str, side: Side, price: float, quantity: int):
        if ticker != "BASKETBALL":
            return
        if side == Side.BUY:
            self.best_bid = price if (self.best_bid is None or price > self.best_bid) else self.best_bid
        else:
            self.best_ask = price if (self.best_ask is None or price < self.best_ask) else self.best_ask

    def on_trade_update(self, trade: Trade):
        if trade.ticker != "BASKETBALL":
            return
        self.recent_trades.append(trade)
        if len(self.recent_trades) > 50:
            self.recent_trades.pop(0)

    def on_game_event_update(self, event: GameEvent):
        self._update_fair_price_advanced(event)
        self._update_momentum(event)
        self._update_quotes_advanced("BASKETBALL", event.time_seconds)

    def _update_fair_price_advanced(self, event: GameEvent):
        old_price = self.fair_price
        score_diff = event.home_score - event.away_score
        time_remaining = event.time_seconds
        time_factor = max(0.01, time_remaining / 2880.0)
        base_prob = 1 / (1 + math.exp(-score_diff * 0.3))
        self.fair_price = base_prob * 100

        event_impact = 0.0
        if event.event_type == "SCORE":
            pts = 3 if event.shot_type == "THREE_POINT" else (1 if event.shot_type == "FREE_THROW" else 2)
            time_weight = 2.0 - time_factor
            event_impact = pts * time_weight * (0.8 if event.home_away == "home" else -0.8)
        elif event.event_type == "MISSED":
            miss_impact = 0.3 * (2.0 - time_factor)
            event_impact = (-miss_impact if event.home_away == "home" else miss_impact)
        elif event.event_type in ["STEAL", "TURNOVER"]:
            momentum_impact = 0.5 * (2.0 - time_factor)
            event_impact = (momentum_impact if event.home_away == "home" else -momentum_impact)

        self.fair_price = max(2.0, min(98.0, self.fair_price + event_impact))

        # volatility proxy
        price_change = abs(self.fair_price - old_price)
        self.volatility = 0.9 * self.volatility + 0.1 * price_change * 5
        self.volatility = max(1.0, min(8.0, self.volatility))

    def _update_momentum(self, event: GameEvent):
        self.momentum *= 0.95
        if event.event_type == "SCORE":
            self.momentum += (1.0 if event.home_away == "home" else -1.0)
        elif event.event_type in ["STEAL", "BLOCK"]:
            self.momentum += (0.5 if event.home_away == "home" else -0.5)
        self.momentum = max(-5.0, min(5.0, self.momentum))

    def _update_quotes_advanced(self, ticker: str, time_seconds: float):
        for oid in self.quote_orders:
            self.cancel_order(ticker, oid)
        self.quote_orders = []

        current_position = self.positions[ticker]
        position_skew = (current_position / self.max_position) * 2.0
        momentum_adj = self.momentum * 0.3
        base_spread = self.volatility * 0.8
        inventory_adj = abs(current_position) / self.max_position * 1.5
        endgame_adj = 1.0 if time_seconds <= 60 else 0.0
        spread = base_spread + inventory_adj + endgame_adj

        mid = self.fair_price + momentum_adj - position_skew
        bid_price = _safe_price(mid - spread / 2)
        ask_price = _safe_price(mid + spread / 2)

        # multi-level quotes
        order_sizes = [100, 75, 50]
        increments = [0.0, 0.3, 0.6]

        if current_position < self.max_position:
            for size, inc in zip(order_sizes, increments):
                level_bid = _safe_price(bid_price - inc)
                if self.capital_remaining > level_bid * size:
                    oid = self.place_limit_order(ticker, Side.BUY, size, level_bid)
                    if oid:
                        self.quote_orders.append(oid)
        if current_position > -self.max_position:
            for size, inc in zip(order_sizes, increments):
                level_ask = _safe_price(ask_price + inc)
                oid = self.place_limit_order(ticker, Side.SELL, size, level_ask)
                if oid:
                    self.quote_orders.append(oid)


class NoiseTrader(TradingAlgorithm):
    """Random trader that adds noise to the market."""

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.trade_probability = 0.05  # 5% chance to trade on each event
        self.max_order_size = 50
        self.max_position = 500

    def on_game_event_update(self, event: GameEvent):
        if random.random() < self.trade_probability:
            self._make_random_trade("BASKETBALL")

    def _make_random_trade(self, ticker: str):
        current_position = self.positions[ticker]

        # Random order parameters
        side = random.choice([Side.BUY, Side.SELL])
        size = random.randint(10, self.max_order_size)

        # Don't trade if it would exceed position limits
        if side == Side.BUY and current_position + size > self.max_position:
            return
        if side == Side.SELL and current_position - size < -self.max_position:
            return

        # Random choice between market and limit order
        if random.random() < 0.3:  # 30% market orders
            self.place_market_order(ticker, side, size)
        else:  # 70% limit orders
            # Random price around current fair value
            base_price = 50 + random.gauss(0, 15)  # Rough estimate
            price = max(1.0, min(99.0, base_price))
            self.place_limit_order(ticker, side, size, price, ioc=True)

# ---------- New Adversary 1: JumpChaserBot ----------
class JumpChaserBot(TradingAlgorithm):
    """
    Crosses aggressively for a few events after high-impact plays (3PT, STEAL, AND-1/FT),
    then cools down. Mimics momentum-chasing flow that widens moves.
    """
    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.chase_strength = 0
        self.cooldown = 0
        self.max_chase = 3
        self.max_position = 400

    def on_game_event_update(self, event: GameEvent):
        if event.event_type == "SCORE" and event.shot_type == "THREE_POINT":
            self.chase_strength = min(self.max_chase, self.chase_strength + 2)
        elif event.event_type == "SCORE" and event.shot_type == "FREE_THROW":
            self.chase_strength = min(self.max_chase, self.chase_strength + 1)
        elif event.event_type in ["STEAL", "TURNOVER", "BLOCK"]:
            self.chase_strength = min(self.max_chase, self.chase_strength + 1)

        action = None
        size = 0
        if self.chase_strength > 0:
            if event.home_away == "home":
                action = Side.BUY
            elif event.home_away == "away":
                action = Side.SELL
            size = 25 + 5 * self.chase_strength

        # respect position caps
        cur = self.positions["BASKETBALL"]
        if action == Side.BUY and cur + size > self.max_position:
            size = max(0, self.max_position - cur)
        if action == Side.SELL and cur - size < -self.max_position:
            size = max(0, cur + self.max_position)

        if action and size > 0:
            self.place_market_order("BASKETBALL", action, size)
            self.chase_strength -= 1

# ---------- New Adversary 2: ContrarianSnapBot ----------
class ContrarianSnapBot(TradingAlgorithm):
    """
    Fades big moves 2-5 events after they happen, betting on partial mean reversion.
    """
    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.queue = []   # (event_time, side_to_fade, size)
        self.max_position = 300

    def on_game_event_update(self, event: GameEvent):
        # schedule a fade a few events later after large scoring plays
        if event.event_type == "SCORE":
            pts = 3 if event.shot_type == "THREE_POINT" else (2 if event.shot_type in ["DUNK","LAYUP","TWO_POINT"] else 1)
            if pts >= 3:
                side = Side.SELL if event.home_away == "home" else Side.BUY
                delay = random.randint(2, 5)
                size = 30 + 10 * (pts - 2)
                self.queue.append((event.time_seconds, side, size, delay))

        # advance queue, execute when delay counts down
        new_queue = []
        for (ts, side, size, delay) in self.queue:
            if delay <= 0:
                cur = self.positions["BASKETBALL"]
                # cap to max inventory
                if side == Side.BUY and cur + size > self.max_position:
                    size = max(0, self.max_position - cur)
                if side == Side.SELL and cur - size < -self.max_position:
                    size = max(0, cur + self.max_position)
                if size > 0:
                    self.place_market_order("BASKETBALL", side, size)
            else:
                new_queue.append((ts, side, size, delay - 1))
        self.queue = new_queue

# ---------- New Adversary 3: EndgameWidenMM ----------
class EndgameWidenMM(TradingAlgorithm):
    """
    Provides two-sided quotes early, then aggressively widens or pulls liquidity
    in the last minute, forcing worse execution for late crossers.
    """
    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.fair_price = 50.0
        self.base_spread = 1.6
        self.order_size = 120
        self.max_position = 900
        self.quote_ids = []

    def on_game_event_update(self, event: GameEvent):
        # fair price = 50 + scaled score diff
        score_diff = event.home_score - event.away_score
        tf = max(0.1, event.time_seconds / 2880.0)
        self.fair_price = max(5.0, min(95.0, 50.0 + score_diff * (1.8 - tf) * 0.5))
        self._quote("BASKETBALL", event.time_seconds)

    def _quote(self, ticker: str, time_seconds: float):
        for oid in self.quote_ids:
            self.cancel_order(ticker, oid)
        self.quote_ids = []

        cur = self.positions[ticker]
        inv_adj = abs(cur) / self.max_position * 2.0
        endgame_widen = 1.2 if time_seconds <= 60 else (0.4 if time_seconds <= 180 else 0.0)
        spread = self.base_spread + inv_adj + endgame_widen

        mid = self.fair_price - (cur / max(1, self.max_position))  # inventory skew
        bid = _safe_price(mid - spread / 2)
        ask = _safe_price(mid + spread / 2)

        # Fewer quotes very late
        levels = 1 if time_seconds <= 60 else 2
        for lvl in range(levels):
            size = max(40, self.order_size - 30 * lvl)
            if cur < self.max_position and self.capital_remaining > bid * size:
                oid = self.place_limit_order(ticker, Side.BUY, size, bid - lvl * 0.2)
                if oid: self.quote_ids.append(oid)
            if cur > -self.max_position:
                oid = self.place_limit_order(ticker, Side.SELL, size, ask + lvl * 0.2)
                if oid: self.quote_ids.append(oid)

# ========================== Human-like Bots ==========================

class PatientValueInvestorBot(TradingAlgorithm):
    """
    Human-like, patient value trader:
      - Ignores early noise (no trading until gate_time).
      - Computes a damped fair value (logistic on score diff) with strong smoothing early.
      - Trades only on big, time-adjusted edges with laddered LIMITs; avoids crossing.
      - Tight risk limits + small 'give-up' stop if adverse move persists.
    """

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        # Behavior knobs
        self.gate_time = 2400.0    # no trading before 8:00 into Q1
        self.target_pos = 300
        self.max_pos = 450
        self.ladder_sizes = [40, 30, 20]
        self.edge_base = 4.0        # $ edge required at mid-game
        self.smooth_halflife = 120  # seconds
        self.stop_giveup = 6.0      # $ adverse vs entry => clear position
        self._fv_ewma = 50.0
        self._last_trade = None
        self._avg_entry = None

    def _logistic_fair(self, score_diff: int, t: float) -> float:
        # Base logistic on score diff; damp early by multiplying slope with (elapsed/total)^0.7
        elapsed = max(0.0, 2880.0 - t)
        time_weight = (elapsed / 2880.0) ** 0.7
        slope = 0.25 + 0.75 * (1.0 - (t / 2880.0))  # steeper late
        slope *= max(0.25, time_weight)             # strong early damping
        p = 1.0 / (1.0 + math.exp(-score_diff * slope))
        return max(2.0, min(98.0, p * 100.0))

    def on_trade_update(self, trade: Trade):
        if trade.ticker != "BASKETBALL":
            return
        self._last_trade = trade.price
        # crude avg entry tracker (position-weighted)
        pos = self.positions["BASKETBALL"]
        if pos != 0:
            if self._avg_entry is None:
                self._avg_entry = trade.price
            else:
                # push towards trade price slightly on fills
                self._avg_entry = 0.95 * self._avg_entry + 0.05 * trade.price
        else:
            self._avg_entry = None

    def on_game_event_update(self, event: GameEvent):
        t = event.time_seconds
        if t > self.gate_time:
            # Just update smoothed fair; don't trade
            raw_fv = self._logistic_fair(event.home_score - event.away_score, t)
            alpha = 1.0 - math.exp(-1.0 / max(1, self.smooth_halflife))  # ~EWMA step
            self._fv_ewma = (1 - alpha) * self._fv_ewma + alpha * raw_fv
            return

        # Compute time-adjusted fair & EWMA smoothing
        raw_fv = self._logistic_fair(event.home_score - event.away_score, t)
        alpha = 1.0 - math.exp(-1.0 / max(1, self.smooth_halflife))
        self._fv_ewma = (1 - alpha) * self._fv_ewma + alpha * raw_fv

        # Mid proxy
        mid = self._last_trade if self._last_trade is not None else self._fv_ewma
        edge = self._fv_ewma - mid

        # Require larger edges early; shrink threshold late
        late_weight = 1.0 - (t / 2880.0)
        need = self.edge_base + 3.0 * max(0.0, 0.5 - late_weight)  # ~more when not very late

        # Manage risk: stop out if far through average entry
        pos = self.positions["BASKETBALL"]
        if self._avg_entry is not None and pos != 0:
            adverse = (mid - self._avg_entry) * (1 if pos > 0 else -1)
            if adverse < -self.stop_giveup:
                # Give up and flatten with IOC
                side = Side.SELL if pos > 0 else Side.BUY
                self.place_limit_order("BASKETBALL", side, abs(pos), mid, ioc=True)
                return

        # Laddered passive accumulation/distribution
        desired = 0
        if edge > need:
            desired = min(self.target_pos, self.max_pos)
        elif edge < -need:
            desired = -min(self.target_pos, self.max_pos)

        delta = desired - pos
        if abs(delta) < 10:
            return

        side = Side.BUY if delta > 0 else Side.SELL
        qty_plan = abs(delta)
        # Place 2-3 passive ladders around mid ± 0.2/0.5
        steps = int(min(len(self.ladder_sizes), max(1, qty_plan // 20)))
        sizes = self.ladder_sizes[:steps]
        for k, sz in enumerate(sizes):
            if sz <= 0:
                continue
            if side == Side.BUY:
                px = max(1.0, min(99.0, mid - (0.2 + 0.3 * k)))
            else:
                px = max(1.0, min(99.0, mid + (0.2 + 0.3 * k)))
            self.place_limit_order("BASKETBALL", side, sz, px)


class ConfirmationMomentumBot(TradingAlgorithm):
    """
    Human-like momentum follower:
      - No trades until gate_time (patience).
      - Needs confirmation: sequence of FV increases + break above MA for buys (sym for sells).
      - Small initial size, pyramids into strength; trailing stop to avoid giving back.
    """

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.gate_time = 2280.0  # ~10 min into game remaining
        self.ma_len = 25
        self._fv_buf = []
        self._fv_ma = 50.0
        self._last_trade = None
        self.base_unit = 20
        self.max_pos = 300
        self.trail = 3.0
        self._peak = None  # for trailing stop reference

    def on_trade_update(self, trade: Trade):
        if trade.ticker == "BASKETBALL":
            self._last_trade = trade.price

    def _update_ma(self, fv: float):
        self._fv_buf.append(fv)
        if len(self._fv_buf) > self.ma_len:
            self._fv_buf.pop(0)
        self._fv_ma = sum(self._fv_buf) / len(self._fv_buf)

    def on_game_event_update(self, event: GameEvent):
        t = event.time_seconds
        # derive a calm fair from score diff; modest slope
        score_diff = event.home_score - event.away_score
        slope = 0.25 + 0.5 * (1.0 - t / 2880.0)  # steeper late, but still modest
        fv = 1.0 / (1.0 + math.exp(-score_diff * slope)) * 100.0
        fv = max(2.0, min(98.0, fv))
        self._update_ma(fv)

        if t > self.gate_time:
            return

        mid = self._last_trade if self._last_trade is not None else fv
        pos = self.positions["BASKETBALL"]

        # Confirmation: 3 rising fv points & fv > MA for longs (opp for shorts)
        confirm_up = len(self._fv_buf) >= 3 and (self._fv_buf[-1] > self._fv_buf[-2] > self._fv_buf[-3]) and (fv > self._fv_ma + 0.6)
        confirm_dn = len(self._fv_buf) >= 3 and (self._fv_buf[-1] < self._fv_buf[-2] < self._fv_buf[-3]) and (fv < self._fv_ma - 0.6)

        # Trailing stop logic
        if pos > 0:
            self._peak = max(self._peak or mid, mid)
            if self._peak - mid >= self.trail:
                # stop out
                self.place_limit_order("BASKETBALL", Side.SELL, pos, mid, ioc=True)
                self._peak = None
                return
        elif pos < 0:
            trough = min(self._peak or mid, mid)
            self._peak = trough
            if mid - self._peak >= self.trail:
                self.place_limit_order("BASKETBALL", Side.BUY, -pos, mid, ioc=True)
                self._peak = None
                return
        else:
            self._peak = mid

        # Entries & pyramiding
        if confirm_up and pos < self.max_pos:
            add = min(self.base_unit, self.max_pos - pos)
            self.place_limit_order("BASKETBALL", Side.BUY, add, max(1.0, mid - 0.2))
        elif confirm_dn and -pos < self.max_pos:
            add = min(self.base_unit, self.max_pos + pos)
            self.place_limit_order("BASKETBALL", Side.SELL, add, min(99.0, mid + 0.2))


class RunAwareContrarianBot(TradingAlgorithm):
    """
    Human-like run fader:
      - Detects short scoring runs and overreaction (esp. early-mid).
      - Schedules fades a few events after a run stalls.
      - Small, spaced entries; clears if move keeps going (soft stop).
    """

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.max_pos = 260
        self._current_run = {"team": None, "length": 0, "pts": 0}
        self._queue = []  # (delay, side, size)
        self._last_mid = None
        self.stop_soft = 5.0

    def on_trade_update(self, trade: Trade):
        if trade.ticker == "BASKETBALL":
            self._last_mid = trade.price

    def on_game_event_update(self, event: GameEvent):
        # Track runs
        if event.event_type == "SCORE":
            pts = 3 if event.shot_type == "THREE_POINT" else 2 if event.shot_type in ["TWO_POINT","DUNK","LAYUP"] else 1
            if self._current_run["team"] == event.home_away:
                self._current_run["length"] += 1
                self._current_run["pts"] += pts
            else:
                # Run flips — schedule a fade of the previous run after a short delay
                if self._current_run["length"] >= 2:
                    # fade previous direction
                    side = Side.SELL if self._current_run["team"] == "home" else Side.BUY
                    size = 15 + 5 * min(4, self._current_run["length"])
                    self._queue.append((3, side, size))  # 3 events later
                self._current_run = {"team": event.home_away, "length": 1, "pts": pts}

        # Process queue (delayed fades)
        new_q = []
        for delay, side, size in self._queue:
            if delay <= 0:
                pos = self.positions["BASKETBALL"]
                # size cap by inventory
                if side == Side.BUY and pos + size > self.max_pos:
                    size = max(0, self.max_pos - pos)
                if side == Side.SELL and pos - size < -self.max_pos:
                    size = max(0, pos + self.max_pos)
                if size > 0:
                    mid = self._last_mid if self._last_mid is not None else 50.0
                    px = max(1.0, min(99.0, mid + (-0.2 if side == Side.BUY else 0.2)))
                    self.place_limit_order("BASKETBALL", side, size, px)
            else:
                new_q.append((delay - 1, side, size))
        self._queue = new_q

        # Soft stop if run continues strongly against us
        if self._last_mid is not None:
            pos = self.positions["BASKETBALL"]
            if pos != 0:
                # Use a crude average ref: if we have losing position beyond stop_soft => flatten
                # (We don't track true avg cost here; we use last mid and assume adverse drift)
                adverse = self.stop_soft  # interpret as threshold for giving up
                # simple heuristic: if run length now >= 4, just clear half
                if self._current_run["length"] >= 4:
                    side = Side.SELL if pos > 0 else Side.BUY
                    self.place_limit_order("BASKETBALL", side, abs(pos)//2, self._last_mid, ioc=True)


class RiskParityMarketMaker(TradingAlgorithm):
    """
    Human-like MM:
      - Scales spread and size with recent volatility ("risk parity" flavor).
      - Smaller, wider quotes during high vol; narrower, larger during calm.
      - Inventory-skewed microprice; pull back late.
    """

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.vol_win = 30
        self._px = []            # recent trade prices
        self._mid = 50.0
        self.max_pos = 700
        self.base_size = 80
        self.quote_ids = []

    def on_trade_update(self, trade: Trade):
        if trade.ticker != "BASKETBALL":
            return
        self._mid = trade.price
        self._px.append(trade.price)
        if len(self._px) > self.vol_win:
            self._px.pop(0)

    def _vol(self) -> float:
        if len(self._px) < 2:
            return 1.5
        diffs = [abs(self._px[i] - self._px[i-1]) for i in range(1, len(self._px))]
        return max(0.5, min(6.0, sum(diffs) / len(diffs)))  # avg absolute change

    def on_game_event_update(self, event: GameEvent):
        # Pull quotes late
        t = event.time_seconds
        late_widen = 1.2 if t <= 90 else (0.4 if t <= 180 else 0.0)

        # Cancel existing
        for oid in self.quote_ids:
            self.cancel_order("BASKETBALL", oid)
        self.quote_ids = []

        v = self._vol()
        # Spread grows with vol; size shrinks with vol
        spread = 1.0 + 0.6 * v + late_widen
        size = max(20, int(self.base_size * (1.4 / (0.6 + v))))  # smaller when v is big

        # Inventory skew
        pos = self.positions["BASKETBALL"]
        inv_skew = (pos / self.max_pos) * 0.8
        mid = self._mid - inv_skew

        # Two levels each side
        for lvl, bump in enumerate([0.0, 0.3]):
            bid = max(1.0, min(99.0, mid - spread/2 - bump))
            ask = max(1.0, min(99.0, mid + spread/2 + bump))
            if pos < self.max_pos and self.capital_remaining > bid * size:
                oid = self.place_limit_order("BASKETBALL", Side.BUY, size, bid)
                if oid: self.quote_ids.append(oid)
            if pos > -self.max_pos:
                oid = self.place_limit_order("BASKETBALL", Side.SELL, size, ask)
                if oid: self.quote_ids.append(oid)
# ====================================================================
