"""
Quant Challenge 2025

Balanced Bayesian + Kelly Strategy (single file)
- Dynamic 4-minute learning guard for ANY game format
- Conservative but active: smaller edges than ultra-conservative so it engages
- Scaled Kelly sizing with hard caps + per-order caps + cooldowns
- Take-profit and soft stop-loss trims
- Spread-aware execution (cross only when reasonable)
"""

from enum import Enum
from typing import Optional
import math, time


# ---- Sandbox template (leave these names/signatures) -----------------------

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Sandbox will route this. Keep signature."""
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Sandbox will route this. Keep signature."""
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Sandbox will route this. Keep signature."""
    return False


# ---- Strategy --------------------------------------------------------------

class Strategy:
    """
    Balanced Bayesian/GAM-lite fair-value + partial-Kelly sizing.
    - Dynamic 4-minute early learning guard (works for NBA/NCAA/etc.)
    - Smooth, conservative sizing and edges so it actually engages
    - Take-profit and soft stop-loss
    - Spread-aware execution & cooldown to keep volume low
    """

    # ===== Risk & pacing knobs (tune here) =====
    # Kelly & sizing
    KELLY_SCALE = 0.04          # fraction of full Kelly (raise to increase activity/size)
    MAX_POSITION_SHARES = 80.0  # hard cap on inventory magnitude
    MAX_ORDER_SHARES = 10.0     # per-order cap (keeps volume/pacing tame)
    TRADE_COOLDOWN_MS = 4000    # min ms between orders

    # Edge thresholds (probability points, in [0,1] space)
    EDGE_ENTRY_EARLY  = 0.060
    EDGE_ENTRY_MID    = 0.050
    EDGE_ENTRY_LATE   = 0.040
    EDGE_ADD          = 0.080   # stronger edge needed to add more aggressively

    # Execution controls
    MAX_CROSS_SPREAD = 1.50     # only cross (market) if spread <= $1.50; else quote passively
    PASSIVE_NUDGE     = 0.10    # improve top-of-book a touch when quoting

    # Profit taking / protection (per-share $)
    TAKE_PROFIT    = 2.25
    SOFT_STOP_LOSS = 3.25

    # Price clamps for safety
    MIN_PRICE = 1.0
    MAX_PRICE = 99.0

    # ===== End of knobs =====

    # ----- Lifecycle -----

    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        # Market state
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.last_trade: Optional[float] = None
        self.mid_price: float = 50.0

        # Inventory / accounting (sandbox will manage true cash; we mirror for references)
        self.position: float = 0.0
        self.cash: float = 100000.0
        self.avg_entry: Optional[float] = None

        # Model state
        self.fair_value: float = 50.0    # our fair value in $ (probability * 100)
        self.momentum: float = 0.0
        self._fair_ema = 50.0
        self._mid_ema  = 50.0
        self._ema_alpha = 0.12
        self.last_order_ms: int = 0

        # Learning guard (dynamic 4-minute guard for any format)
        self.game_start_time_seconds: Optional[float] = None
        self.learn_until: Optional[float] = None   # = start_time_seconds - 240.0

    # ----- Small utilities -----

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _update_mid(self):
        if self.best_bid is not None and self.best_ask is not None:
            self.mid_price = (self.best_bid + self.best_ask) / 2.0
        elif self.last_trade is not None:
            self.mid_price = self.last_trade
        # smooth to avoid overreacting
        self._mid_ema = (1 - self._ema_alpha) * self._mid_ema + self._ema_alpha * self.mid_price
        self.mid_price = self._mid_ema

    def _relative_phase_edge(self, time_seconds: Optional[float]) -> float:
        """Pick edge threshold by *relative* game phase (works for any game length)."""
        if self.game_start_time_seconds is None or time_seconds is None:
            return self.EDGE_ENTRY_MID
        frac = time_seconds / max(1.0, self.game_start_time_seconds)  # >1 early (if start unknown), ~1 at tip
        # time_seconds counts down; so early means frac close to 1
        if frac > 0.70:
            return self.EDGE_ENTRY_EARLY
        elif frac > 0.40:
            return self.EDGE_ENTRY_MID
        else:
            return self.EDGE_ENTRY_LATE

    # ----- Fair value (Bayesian-flavored, momentum & time scaling) -----

    def _compute_fair(self, home: int, away: int, t_secs: Optional[float]) -> float:
        """Logistic on score diff scaled by time, adjusted by decaying momentum."""
        if t_secs is None:
            t_secs = 1440.0  # fallback
        # core signal: score differential weighted more as time declines
        lead = (home - away) / math.sqrt(max(1.0, t_secs)) * 2.6
        # time-of-game weight (late game => bigger momentum influence)
        time_weight = 1.5 - 0.5 * (t_secs / max(1.0, self.game_start_time_seconds or 2880.0))
        mterm = self.momentum * 0.06 * time_weight
        p = self._sigmoid(lead + mterm)
        fv = self._clip(p * 100.0, self.MIN_PRICE, self.MAX_PRICE)
        # smooth
        self._fair_ema = (1 - self._ema_alpha) * self._fair_ema + self._ema_alpha * fv
        return self._fair_ema

    # ----- Kelly sizing (scaled) -----

    def _kelly_target_shares(self, p_hat: float, s_prob: float, price_dollars: float, cash: float) -> float:
        """
        p_hat : our win prob in [0,1]
        s_prob: market-implied prob in [0,1] (mid/100)
        price_dollars: approx contract price
        """
        p = self._clip(p_hat, 1e-4, 1 - 1e-4)
        s = self._clip(s_prob, 1e-4, 1 - 1e-4)
        edge = p - s
        # Kelly fraction for binary payoff ~ edge / variance proxy
        if edge > 0:
            f = edge / (1.0 - s)
        else:
            f = edge / (s)
        f *= self.KELLY_SCALE
        # Convert to shares using cash & price
        target = (f * cash) / max(1.0, price_dollars)
        return self._clip(target, -self.MAX_POSITION_SHARES, self.MAX_POSITION_SHARES)

    # ----- Trade mgmt helpers -----

    def _update_avg_entry_after_fill(self, side: Side, qty: float, price: float):
        """Maintain a reasonable average entry reference for TP/SL."""
        pos0 = self.position
        if side == Side.BUY:
            # increasing long or reducing short
            if pos0 >= 0:
                # build/extend long
                new_pos = pos0 + qty
                self.avg_entry = price if pos0 == 0 or self.avg_entry is None else \
                    (self.avg_entry * pos0 + price * qty) / max(1.0, new_pos)
            else:
                # reducing short
                new_pos = pos0 + qty
                if new_pos >= 0:
                    # flipped from short to flat/long -> reset anchor to price
                    self.avg_entry = price
                # else remain short: keep previous anchor
        else:
            # SELL: increasing short or reducing long
            if pos0 <= 0:
                # build/extend short
                new_pos = pos0 - qty
                self.avg_entry = price if pos0 == 0 or self.avg_entry is None else \
                    (self.avg_entry * (-pos0) + price * qty) / max(1.0, -new_pos)
            else:
                # reducing long
                new_pos = pos0 - qty
                if new_pos <= 0:
                    # flipped from long to flat/short -> reset anchor to price
                    self.avg_entry = price
                # else remain long: keep existing anchor

    def _maybe_take_profit_or_stop(self) -> bool:
        if self.position == 0 or self.mid_price is None or self.avg_entry is None:
            return False
        pnl_ps = (self.mid_price - self.avg_entry) if self.position > 0 else (self.avg_entry - self.mid_price)
        if pnl_ps >= self.TAKE_PROFIT:
            qty = min(abs(self.position), self.MAX_ORDER_SHARES)
            place_market_order(Side.SELL if self.position > 0 else Side.BUY, Ticker.TEAM_A, qty)
            return True
        if pnl_ps <= -self.SOFT_STOP_LOSS:
            qty = min(abs(self.position), self.MAX_ORDER_SHARES)
            place_market_order(Side.SELL if self.position > 0 else Side.BUY, Ticker.TEAM_A, qty)
            return True
        return False

    # ----- Sandbox callbacks -----

    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != Ticker.TEAM_A:
            return
        self.last_trade = price
        self._update_mid()

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != Ticker.TEAM_A:
            return
        if side == Side.BUY:
            self.best_bid = price if (self.best_bid is None or price > self.best_bid) else self.best_bid
        else:
            self.best_ask = price if (self.best_ask is None or price < self.best_ask) else self.best_ask
        self._update_mid()

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        if ticker != Ticker.TEAM_A:
            return
        # Update average entry & position snapshot for local TP/SL logic
        self._update_avg_entry_after_fill(side, quantity, price)
        signed = quantity if side == Side.BUY else -quantity
        self.position += signed
        self.cash = capital_remaining  # FYI; sandbox is source of truth

        # Clamp to hard cap in case of partial fills racing
        if abs(self.position) > self.MAX_POSITION_SHARES:
            # try to revert excess safely next tick via TP/SL or cooldown gating
            self.position = math.copysign(self.MAX_POSITION_SHARES, self.position)

    def on_game_event_update(
        self,
        event_type: str,
        home_away: str,
        home_score: int,
        away_score: int,
        player_name: Optional[str],
        substituted_player_name: Optional[str],
        shot_type: Optional[str],
        assist_player: Optional[str],
        rebound_type: Optional[str],
        coordinate_x: Optional[float],
        coordinate_y: Optional[float],
        time_seconds: Optional[float]
    ) -> None:

        # Initialize dynamic 4-minute guard when we see the first timestamp
        if self.game_start_time_seconds is None and time_seconds is not None:
            self.game_start_time_seconds = float(time_seconds)
            self.learn_until = max(0.0, self.game_start_time_seconds - 240.0)  # 4 minutes = 240s

        # Update a small momentum (decays naturally)
        self.momentum *= 0.96
        if event_type == "SCORE":
            self.momentum += (1.0 if home_away == "home" else -1.0)
        elif event_type in ("STEAL", "BLOCK"):
            self.momentum += (0.6 if home_away == "home" else -0.6)
        elif event_type == "TURNOVER":
            self.momentum += (-0.4 if home_away == "home" else 0.4)

        # Refresh fair value continuously
        self.fair_value = self._compute_fair(home_score, away_score, time_seconds)

        # Dynamic 4-minute learning guard (works for any game length)
        if time_seconds is None or (self.learn_until is not None and time_seconds > self.learn_until):
            if event_type == "END_GAME":
                self.reset_state()
            return

        # End-game housekeeping
        if event_type == "END_GAME":
            self.reset_state()
            return

        # Try opportunistic trims first (TP/SL), throttle with cooldown
        if self._maybe_take_profit_or_stop():
            self.last_order_ms = self._now_ms()

        # Decide whether to move toward Kelly target
        self._update_mid()
        mid = self._clip(self.mid_price, self.MIN_PRICE, self.MAX_PRICE)

        # Need a visible spread to act intelligently
        spread = None
        if self.best_bid is not None and self.best_ask is not None:
            spread = max(0.0, self.best_ask - self.best_bid)

        # Market implied probability from mid
        p_mkt = self._clip(mid / 100.0, 1e-3, 1 - 1e-3)
        p_hat = self._clip(self.fair_value / 100.0, 1e-3, 1 - 1e-3)
        edge = p_hat - p_mkt
        thr = self._relative_phase_edge(time_seconds)

        # Respect cooldown
        if self._now_ms() - self.last_order_ms < self.TRADE_COOLDOWN_MS:
            return

        # Only engage if the edge is meaningful
        if abs(edge) < thr:
            return

        # Compute target inventory via scaled Kelly
        target = self._kelly_target_shares(p_hat, p_mkt, mid, self.cash)

        # Require stronger edge to add a lot; otherwise glide toward target
        if (abs(edge) < self.EDGE_ADD) and (abs(target) > abs(self.position)):
            target = self.position + (target - self.position) * 0.5

        # Step a small increment toward target (volume control)
        delta = target - self.position
        if abs(delta) < 1.0:
            return
        qty = min(self.MAX_ORDER_SHARES, abs(delta))
        if qty <= 0:
            return

        # Side to move inventory toward target
        side = Side.BUY if delta > 0 else Side.SELL

        # Execution: cross only if spread is reasonable; else join/improve the book
        crossed = False
        if spread is not None and spread <= self.MAX_CROSS_SPREAD:
            place_market_order(side, Ticker.TEAM_A, qty)
            crossed = True
        else:
            # Passive price around top-of-book with a tiny nudge
            if side == Side.BUY:
                px = mid - self.PASSIVE_NUDGE
                if self.best_bid is not None:
                    px = max(self.best_bid + self.PASSIVE_NUDGE, self.MIN_PRICE)
                px = self._clip(px, self.MIN_PRICE, mid)  # don't pay above mid when joining bid
                place_limit_order(Side.BUY, Ticker.TEAM_A, qty, px, ioc=False)
            else:
                px = mid + self.PASSIVE_NUDGE
                if self.best_ask is not None:
                    px = min(self.best_ask - self.PASSIVE_NUDGE, self.MAX_PRICE)
                px = self._clip(px, mid, self.MAX_PRICE)  # don't sell below mid when joining ask
                place_limit_order(Side.SELL, Ticker.TEAM_A, qty, px, ioc=False)

        if crossed:
            self.last_order_ms = self._now_ms()

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        # Optional: could sanity check or reset best bid/ask if needed.
        return