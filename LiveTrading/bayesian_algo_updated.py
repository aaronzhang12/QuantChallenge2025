"""
Quant Challenge 2025

Bayesian-Kelly Strategy (single file)
- Early-game guard for price learning
- Bayesian linear regression (neutral priors) on game & microstructure features
- Kelly criterion sizing adapted for prediction markets (scaled, capped)
- Per-trade & rolling-window notional throttles
"""

from enum import Enum
from typing import Optional, List, Tuple


# === Platform enums & order functions (as provided by sandbox) ===
class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    return 0


# =============================== Strategy ===============================
class Strategy:
    """
    Predict fair price via Bayesian linear regression:
        y_t = x_t · μ   (price in $)
    with posterior updates on each trade print. Then size positions using
    **scaled Kelly** for prediction markets:
        Long fraction  f_L = (π - p)/(1 - p)
        Short fraction f_S = (p - π)/p
    where π = predicted prob (price/100) and p = market prob (mid/100).

    We add:
      - conservative **early-game guard** (learn before risking),
      - **per-trade** and **rolling-window** notional/size caps,
      - inventory-aware quoting when edge is small.
    """

    # ======= Risk/behavior knobs (tune here) =======
    # Early-game: ignore directional risk until N seconds have elapsed,
    # unless a very wide early lead appears.
    MIN_ELAPSED_SEC: int  = 600     # ~10 minutes
    EARLY_WIDE_LEAD: int  = 12      # allow earlier risk if |lead| >= this

    # Edge threshold in dollars between prediction and market (time-decaying).
    BASE_EDGE: float      = 2.0

    # Kelly scaling (0.0..1.0). 0.25 = "quarter Kelly" (human risk tolerance).
    KELLY_SCALE: float    = 0.25

    # Caps to avoid going "all-in"
    MAX_POSITION_SHARES: float = 20000.0      # absolute inventory cap
    MAX_TRADE_SHARES: float    = 2000.0       # per-trade cap
    DEFAULT_CAPITAL: float     = 100000.0     # used until first account update
    MAX_NOTIONAL_PER_120S: float = 40000.0    # throttle: ~$ traded over 120s

    # Maker behavior when we are learning or edge is small
    MAKER_SIZE: float     = 300.0
    MAKER_HALFSPREAD: float = 1.2             # around prediction ($)
    IOC_QUOTES: bool      = True              # post as IOC to avoid clutter

    # Model hyperparameters
    NOISE_VAR: float      = 9.0               # observation variance ($^2)
    PRIOR_VAR: float      = 400.0             # diagonal prior variance

    # Market limits
    MIN_PRICE: float      = 1.0
    MAX_PRICE: float      = 99.0
    TICK: float           = 0.1

    # Feature schema
    _EVENTS = ("SCORE","STEAL","TURNOVER","BLOCK","FOUL","MISSED",
               "SUBSTITUTION","TIMEOUT","END_PERIOD","END_GAME")
    _SHOTS  = ("THREE_POINT","TWO_POINT","DUNK","LAYUP","FREE_THROW")
    # d = 1 (bias) + 2 (time) + 4 (score) + 3 (momentum) + 2 (run/pace)
    #   + 10 (events) + 5 (shots) + 2 (micro)
    _D = 1 + 2 + 4 + 3 + 2 + len(_EVENTS) + len(_SHOTS) + 2

    def __init__(self) -> None:
        self.reset_state()

    # ---------------- state & helpers ----------------
    def reset_state(self) -> None:
        # Posterior mean/cov (lists; no numpy)
        self.mu: List[float] = [0.0]*self._D
        self.mu[0] = 50.0                          # neutral prior
        self.Sigma: List[List[float]] = [[0.0]*self._D for _ in range(self._D)]
        for i in range(self._D):
            self.Sigma[i][i] = self.PRIOR_VAR

        # Microstructure
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.last_trade: Optional[float] = None

        # Position & capital tracking
        self.position: float = 0.0
        self.capital: float  = self.DEFAULT_CAPITAL

        # Event buffers
        self._score_diffs: List[float] = []
        self._events: List[str] = []
        self._home_aways: List[str] = []

        # For BLR update
        self._last_x: Optional[List[float]] = None

        # Quote housekeeping
        self._resting: List[int] = []

        # Throttle bookkeeping: list of (time_seconds, notional_abs)
        self._recent_notional: List[Tuple[float, float]] = []

    # ---- tiny LA utilities ----
    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(ai*bi for ai,bi in zip(a,b))

    @staticmethod
    def _mat_vec(M: List[List[float]], v: List[float]) -> List[float]:
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

    @staticmethod
    def _outer(u: List[float], v: List[float]) -> List[List[float]]:
        return [[ui*vj for vj in v] for ui in u]

    @staticmethod
    def _mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        m = len(A); n = len(A[0]) if m else 0
        out = [[0.0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                out[i][j] = A[i][j] - B[i][j]
        return out

    # ---- Bayesian update: (x,y) with partial forgetting ----
    def _bayes_update(self, x: List[float], y: float, lr: float = 0.9) -> None:
        Sx = self._mat_vec(self.Sigma, x)                 # Σx
        denom = self.NOISE_VAR + self._dot(x, Sx)         # σ² + xᵀΣx
        if denom <= 1e-9:
            return
        K = [lr * (v/denom) for v in Sx]                  # gain
        resid = y - self._dot(self.mu, x)                 # y - ŷ
        for i in range(self._D):
            self.mu[i] += K[i] * resid                    # μ ← μ + K r
        self.Sigma = self._mat_sub(self.Sigma, self._outer(K, Sx))  # Σ ← Σ - K(Σx)ᵀ

    # ---- features ----
    def _features(self, event_type: str, home_away: str,
                  home_score: int, away_score: int,
                  shot_type: Optional[str], time_seconds: Optional[float]) -> List[float]:
        t = float(time_seconds or 0.0)
        if t < 0.0: t = 0.0
        if t > 2880.0: t = 2880.0
        t_norm = t/2880.0
        t_left = 1.0 - t_norm

        sd = float(home_score - away_score)
        ad = abs(sd)
        sd_t = sd * t_left
        ad_t = ad * t_left

        self._score_diffs.append(sd)
        if len(self._score_diffs) > 120: self._score_diffs.pop(0)
        self._events.append(event_type or "")
        if len(self._events) > 120: self._events.pop(0)
        self._home_aways.append(home_away or "")
        if len(self._home_aways) > 120: self._home_aways.pop(0)

        sd_chg_1  = 0.0 if len(self._score_diffs) < 2  else (self._score_diffs[-1] - self._score_diffs[-2])
        sd_mom_10 = 0.0 if len(self._score_diffs) < 11 else (self._score_diffs[-1] - self._score_diffs[-11])
        sd_mom_30 = 0.0 if len(self._score_diffs) < 31 else (self._score_diffs[-1] - self._score_diffs[-31])

        run_len = 0.0
        if self._events and self._events[-1] == "SCORE":
            team = self._home_aways[-1]
            run_len = 1.0
            j = len(self._events) - 2
            while j >= 0 and self._events[j] == "SCORE" and self._home_aways[j] == team:
                run_len += 1.0
                j -= 1

        pace_60 = float(sum(1 for e in self._events[-60:] if e == "SCORE"))

        ev_oh = [1.0 if event_type == e else 0.0 for e in self._EVENTS]
        sh_oh = [1.0 if (shot_type == s) else 0.0 for s in self._SHOTS]

        spread, mid_dev = 0.0, 0.0
        if self.best_bid is not None and self.best_ask is not None:
            spread = abs(self.best_ask - self.best_bid)
            mid = 0.5*(self.best_bid + self.best_ask)
            last = self.last_trade if self.last_trade is not None else mid
            mid_dev = last - mid

        x: List[float] = []
        x.append(1.0)                      # bias
        x += [t_norm, t_left]              # 2
        x += [sd, ad, sd_t, ad_t]          # 4
        x += [sd_chg_1, sd_mom_10, sd_mom_30]  # 3
        x += [run_len, pace_60]            # 2
        x += ev_oh                         # 10
        x += sh_oh                         # 5
        x += [spread, mid_dev]             # 2

        # Pad guard
        if len(x) < self._D:
            x += [0.0]*(self._D - len(x))
        elif len(x) > self._D:
            x = x[:self._D]
        return x

    # ---- misc ----
    def _clip_price(self, p: float) -> float:
        if p < self.MIN_PRICE: p = self.MIN_PRICE
        if p > self.MAX_PRICE: p = self.MAX_PRICE
        return round(p / self.TICK) * self.TICK

    def _market_mid(self) -> float:
        if self.best_bid is not None and self.best_ask is not None:
            return 0.5*(self.best_bid + self.best_ask)
        if self.last_trade is not None:
            return self.last_trade
        return 50.0

    def _edge_threshold(self, time_seconds: float) -> float:
        # Decay threshold from BASE early -> ~0.6*BASE late
        t = max(0.0, min(2880.0, time_seconds))
        t_norm = t/2880.0
        return self.BASE_EDGE * (0.6 + 0.4*t_norm)

    def _cancel_resting(self) -> None:
        if not self._resting:
            return
        for oid in self._resting:
            try:
                cancel_order(Ticker.TEAM_A, oid)
            except Exception:
                pass
        self._resting = []

    def _throttle_ok(self, time_seconds: float, new_notional: float) -> bool:
        """Keep notional over the last 120s under a cap."""
        lookback = 120.0
        # purge outside window
        self._recent_notional = [(t, n) for (t, n) in self._recent_notional if (t - time_seconds) <= lookback]
        window_sum = sum(n for _, n in self._recent_notional)
        return (window_sum + abs(new_notional)) <= self.MAX_NOTIONAL_PER_120S

    # ================= Platform callbacks =================
    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != Ticker.TEAM_A:
            return
        self.last_trade = float(price)
        if self._last_x is not None:
            self._bayes_update(self._last_x, self.last_trade, lr=0.9)

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != Ticker.TEAM_A:
            return
        if side == Side.BUY:
            if self.best_bid is None or price > self.best_bid:
                self.best_bid = float(price)
        else:
            if self.best_ask is None or price < self.best_ask:
                self.best_ask = float(price)

    def on_account_update(self, ticker: Ticker, side: Side, price: float, quantity: float, capital_remaining: float) -> None:
        if ticker != Ticker.TEAM_A:
            return
        if side == Side.BUY:
            self.position += float(quantity)
        else:
            self.position -= float(quantity)
        self.capital = float(capital_remaining)

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        # Optional: could recompute bests from snapshot
        pass

    def on_game_event_update(self,
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
                             time_seconds: Optional[float]) -> None:

        # End game: clean slate
        if event_type == "END_GAME":
            self._cancel_resting()
            self.reset_state()
            return

        t = float(time_seconds or 0.0)
        x = self._features(event_type, home_away, home_score, away_score, shot_type, t)
        self._last_x = x

        # Prediction → price & probability
        pred_price = self._clip_price(self._dot(self.mu, x))
        mid = self._market_mid()
        edge = pred_price - mid                    # $ edge
        p_mkt = max(0.01, min(0.99, mid / 100.0))  # market prob
        p_hat = max(0.01, min(0.99, pred_price / 100.0))  # model prob

        # Early-game guard (learn first)
        elapsed = 2880.0 - t
        early = elapsed < self.MIN_ELAPSED_SEC
        wide_early = abs(home_score - away_score) >= self.EARLY_WIDE_LEAD

        # Time-decaying edge threshold
        thr = self._edge_threshold(t)

        # If small edge or early (w/o wide lead): quote small two-sided IOC near prediction
        if (abs(edge) < thr) or (early and not wide_early):
            self._cancel_resting()
            half = self.MAKER_HALFSPREAD
            bid = self._clip_price(pred_price - half)
            ask = self._clip_price(pred_price + half)

            if self.position < self.MAX_POSITION_SHARES:
                oid_b = place_limit_order(Side.BUY, Ticker.TEAM_A, self.MAKER_SIZE, bid, ioc=self.IOC_QUOTES)
                if oid_b: self._resting.append(oid_b)
            if self.position > -self.MAX_POSITION_SHARES:
                oid_a = place_limit_order(Side.SELL, Ticker.TEAM_A, self.MAKER_SIZE, ask, ioc=self.IOC_QUOTES)
                if oid_a: self._resting.append(oid_a)
            return

        # ========= Kelly sizing (scaled & capped) =========
        # Long fraction:  f_L = (π - p)/(1 - p)
        # Short fraction: f_S = (p - π)/p
        f_long  = (p_hat - p_mkt) / (1.0 - p_mkt)
        f_short = (p_mkt - p_hat) / max(1e-6, p_mkt)

        # Choose side with positive fraction; scale by KELLY_SCALE
        side: Optional[Side] = None
        f_star: float = 0.0
        if f_long > f_short and f_long > 0.0:
            side = Side.BUY
            f_star = f_long
        elif f_short > 0.0:
            side = Side.SELL
            f_star = f_short

        if side is None or f_star <= 0.0:
            # No Kelly-justified trade → act as maker
            self._cancel_resting()
            half = self.MAKER_HALFSPREAD
            bid = self._clip_price(pred_price - half)
            ask = self._clip_price(pred_price + half)
            if self.position < self.MAX_POSITION_SHARES:
                oid_b = place_limit_order(Side.BUY, Ticker.TEAM_A, self.MAKER_SIZE, bid, ioc=self.IOC_QUOTES)
                if oid_b: self._resting.append(oid_b)
            if self.position > -self.MAX_POSITION_SHARES:
                oid_a = place_limit_order(Side.SELL, Ticker.TEAM_A, self.MAKER_SIZE, ask, ioc=self.IOC_QUOTES)
                if oid_a: self._resting.append(oid_a)
            return

        # Scaled Kelly desired notional
        kelly_scaled = max(0.0, min(1.0, self.KELLY_SCALE * f_star))
        capital = max(self.capital, self.DEFAULT_CAPITAL * 0.5)  # fallback until first fill
        desired_notional = kelly_scaled * capital                 # dollars to deploy
        px = max(self.MIN_PRICE, min(self.MAX_PRICE, mid))
        desired_shares = desired_notional / px
        if side == Side.SELL:
            desired_shares = -desired_shares

        # Convert to target inventory & delta
        target_pos = desired_shares
        # Clamp to global position limits
        if target_pos > self.MAX_POSITION_SHARES:  target_pos = self.MAX_POSITION_SHARES
        if target_pos < -self.MAX_POSITION_SHARES: target_pos = -self.MAX_POSITION_SHARES

        delta = target_pos - self.position
        if abs(delta) < 1.0:
            # Already close enough → rest as maker
            return

        # Per-trade cap & rolling notional throttle
        qty = max(10.0, min(self.MAX_TRADE_SHARES, abs(delta)))
        notional = qty * px
        if not self._throttle_ok(t, notional):
            # Skip crossing; throttle exceeded → passively quote instead
            self._cancel_resting()
            half = self.MAKER_HALFSPREAD
            bid = self._clip_price(pred_price - half)
            ask = self._clip_price(pred_price + half)
            if self.position < self.MAX_POSITION_SHARES:
                oid_b = place_limit_order(Side.BUY, Ticker.TEAM_A, self.MAKER_SIZE, bid, ioc=self.IOC_QUOTES)
                if oid_b: self._resting.append(oid_b)
            if self.position > -self.MAX_POSITION_SHARES:
                oid_a = place_limit_order(Side.SELL, Ticker.TEAM_A, self.MAKER_SIZE, ask, ioc=self.IOC_QUOTES)
                if oid_a: self._resting.append(oid_a)
            return

        # Urgency: stronger edge & later game → more aggressive
        urgency = (abs(edge) - thr) / max(thr, 1.0) + (1.0 - t/2880.0) * 0.5
        self._cancel_resting()

        if urgency > 0.8:
            place_market_order(Side.BUY if delta > 0 else Side.SELL, Ticker.TEAM_A, qty)
        else:
            # Work a limit near prediction to reduce slippage
            lim = pred_price - 0.3 if delta > 0 else pred_price + 0.3
            lim = self._clip_price(lim)
            place_limit_order(Side.BUY if delta > 0 else Side.SELL, Ticker.TEAM_A, qty, lim, ioc=False)

        # Record notional for throttle window
        self._recent_notional.append((t, notional))