"""
Quant Challenge 2025

GAM-with-splines Bayesian strategy with online learning + Kelly sizing
Single-file submission (no external assets).
"""

from enum import Enum
from typing import Optional, List

# ---- Template-provided types ----
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

# =============== Strategy ===============

class Strategy:
    """GAM-based probability + online learning + Kelly sizing.

    - Offline model: piecewise-linear splines (hinge basis) trained on logs.
    - Online learning: stochastic Newton/gradient step to track regime shifts using smoothed mid price as a proxy.
    - Early-learning guard: no directional risk in the first few minutes unless the lead is unusually large.
    - Sizing: Kelly fraction with a safety cap; rebalance toward target gradually.

    The model outputs the home win probability p \in [0,1]. All prices assumed in [0,100].
    """

    # ---- Offline model payload (baked-in) ----
    _BETA = [0.6698712484772685, -0.8459477305599291, -1.0775474969477084, -0.08506466176089428, -0.21154819724474372, -0.40355786859597004, -0.037298128846283065, 0.6208979935013876, 0.6508082711781566, 0.38263356478894964, -0.0947924978128198, 0.4627100012361267, 0.4432660328264812, 0.3566530711997396, 0.8226540081548164, 0.7487099080358187, 0.577009435634836, 0.026634849548100618, 0.40607701917095625, 0.5139594354977383, 0.17278609294384574, 0.01869708767890363, -0.039725853122084286, -0.011120275814108035, -0.01383236270778067, -0.006251685580237483, 0.39931246425364607, -0.010079583518792548, 0.02383223242069353, -0.014117738238164901, 0.011140748204897237, 0.0004013572723124809, -0.012994891727468785, -0.006651991989363237]
    _T_KNOTS = [0.03979166666666667, 0.159375, 0.31875, 0.47708333333333336, 0.6354166666666666, 0.79375]         # on t_norm = time_seconds / 2880
    _SD_KNOTS = [-2.0, 0.0, 1.0, 3.0, 5.0, 8.0]       # on score_diff = home - away
    _ABS_KNOTS = [0.0, 1.0, 4.0]     # on |score_diff|
    _MOM_KNOTS = [-1.6079000000000002, -0.6053819999999999, -0.09366799999999993, 0.4666280000000001, 2.0732100000000004]     # on momentum (decayed)
    _EVENTS = ['SCORE', 'STEAL', 'TURNOVER', 'BLOCK', 'FOUL', 'TIMEOUT', 'SUBSTITUTION', 'END_PERIOD']           # one-hots in this order

    # ---- Hyperparameters (safe defaults) ----
    EARLY_GUARD_SECONDS = 360         # first 6 minutes = learn market, avoid overreaction
    EARLY_LEAD_THRESH_START = 8       # if lead >= 8 early, allow trading; shrinks linearly during guard
    EARLY_LEAD_THRESH_END = 4
    MID_EMA_ALPHA = 0.25              # smoothing for observed mid price
    ONLINE_L2 = 5.0                   # ridge penalty in online update
    ONLINE_STEP = 0.6                 # step size for online update (0..1), conservative
    KELLY_CAP = 0.1                   # use at most 25% Kelly
    MAX_NOTIONAL = 50000.0            # cap absolute notional exposure (USD)
    REBALANCE_FRACTION = 0.35         # move this fraction toward target per event
    MIN_EDGE_BP = 40                  # ignore edges smaller than 40 bps (0.40%)
    MIN_TRADE_QTY = 5.0               # avoid micro-churn
    MAX_TRADE_QTY = 200.0             # per action throttle

    def reset_state(self) -> None:
        # Internal book-keeping
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.mid_smooth: Optional[float] = None

        self.position: float = 0.0
        self.capital_remaining: float = 100000.0

        # momentum & last features
        self.momentum: float = 0.0
        self.last_features: Optional[List[float]] = None

        # model params (mutable for online learning)
        self.beta = list(self._BETA)

    def __init__(self) -> None:
        self.reset_state()

    # ---------- Utility: basis construction ----------
    @staticmethod
    def _relu(x: float) -> float:
        return x if x > 0.0 else 0.0

    @classmethod
    def _hinge_vector(cls, x: float, knots: List[float]) -> List[float]:
        # [x, (x-k1)_+, ..., (x-km)_+]
        v = [x]
        for k in knots:
            v.append(cls._relu(x - k))
        return v

    def _feature_vector(self,
                        event_type: str,
                        home_away: Optional[str],
                        home_score: Optional[int],
                        away_score: Optional[int],
                        shot_type: Optional[str],
                        time_seconds: Optional[float],
                        points: int) -> List[float]:
        # Guard against None
        hs = float(home_score or 0)
        as_ = float(away_score or 0)
        sd = hs - as_
        t = float(time_seconds or 0.0)
        t_norm = max(0.0, min(1.0, t/2880.0))

        # basis blocks
        feat = [1.0]  # intercept

        feat += self._hinge_vector(t_norm, self._T_KNOTS)
        feat += self._hinge_vector(sd, self._SD_KNOTS)
        feat += self._hinge_vector(abs(sd), self._ABS_KNOTS)
        feat += self._hinge_vector(self.momentum, self._MOM_KNOTS)

        # immediate points from this event
        feat.append(float(points))

        # event one-hots in fixed order
        ev = event_type or ""
        for name in self._EVENTS:
            feat.append(1.0 if ev == name else 0.0)

        return feat  # length must match len(beta)

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = pow(2.718281828459045, -z)
            return 1.0/(1.0+ez)
        else:
            ez = pow(2.718281828459045, z)
            return ez/(1.0+ez)

    def _predict_prob(self, x: List[float]) -> float:
        z = 0.0
        for b, xi in zip(self.beta, x):
            z += b*xi
        p = self._sigmoid(z)
        # clamp numerical safety
        if p < 1e-4: p = 1e-4
        if p > 1-1e-4: p = 1-1e-4
        return p

    # ---------- Online update ----------
    def _online_update(self, x: List[float], y_target_prob: float) -> None:
        """One-step IRLS/gradient update to gently nudge beta toward price."""
        # prediction
        z = sum(b*xi for b, xi in zip(self.beta, x))
        p = self._sigmoid(z)
        # weights and gradient (ridge)
        w = p*(1-p) + 1e-6
        # Newton step on a single observation (approximate)
        # beta_new = beta + step * ( (y-p)/w * x ) / (x^2 + l2)
        l2 = self.ONLINE_L2
        step = self.ONLINE_STEP
        # denom ~ x^T W x + l2 with scalar W
        denom = l2
        for xi in x:
            denom += (xi*xi) * w
        scale = step * ( (y_target_prob - p) / max(1e-6, denom) )
        # update
        for i in range(len(self.beta)):
            self.beta[i] += scale * x[i]

    # ---------- Sizing (Kelly) ----------
    def _kelly_target_notional(self, p_belief: float) -> float:
        m = self._mid()
        if m is None:
            return 0.0
        pm = m/100.0  # market prob
        p = max(1e-4, min(1-1e-4, p_belief))

        edge = p - pm
        # 40 bps edge threshold on probability scale
        if abs(edge) < self.MIN_EDGE_BP * 1e-4:
            return 0.0

        # Kelly fraction of bankroll to allocate in the direction of edge
        if edge > 0:
            # buy contracts
            f_star = edge / (1.0 - pm)
            direction = 1.0
        else:
            # sell contracts (short)
            f_star = (-edge) / max(pm, 1e-4)
            direction = -1.0

        f_star = max(0.0, min(f_star, 1.0))
        f_star *= self.KELLY_CAP

        bankroll = self.capital_remaining
        target_notional = direction * f_star * bankroll
        # cap absolute notional
        if abs(target_notional) > self.MAX_NOTIONAL:
            target_notional = self.MAX_NOTIONAL * (1.0 if target_notional > 0 else -1.0)
        return target_notional

    # ---------- Helpers ----------
    def _mid(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return 0.5*(self.best_bid + self.best_ask)
        return None

    def _lead_threshold(self, elapsed_s: float) -> float:
        if elapsed_s >= self.EARLY_GUARD_SECONDS:
            return 0.0
        # linear interpolation from start->end
        frac = max(0.0, min(1.0, elapsed_s / self.EARLY_GUARD_SECONDS))
        return self.EARLY_LEAD_THRESH_START + frac*(self.EARLY_LEAD_THRESH_END - self.EARLY_LEAD_THRESH_START)

    # ---------- Callbacks ----------
    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        # pass-through; mid can be nudged by last trade if no quotes
        if price and self.mid_smooth is None:
            self.mid_smooth = float(price)

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != Ticker.TEAM_A:
            return
        if side == Side.BUY:
            if self.best_bid is None or price > self.best_bid:
                self.best_bid = price
        else:
            if self.best_ask is None or price < self.best_ask:
                self.best_ask = price

        # Update smoothed mid when both sides present
        m = self._mid()
        if m is not None:
            if self.mid_smooth is None:
                self.mid_smooth = m
            else:
                a = self.MID_EMA_ALPHA
                self.mid_smooth = a*m + (1-a)*self.mid_smooth

        # Perform an online parameter update if we have last features
        if self.last_features is not None and self.mid_smooth is not None:
            y_target = max(0.01, min(0.99, self.mid_smooth/100.0))
            self._online_update(self.last_features, y_target)

    def on_account_update(self,
                          ticker: Ticker,
                          side: Side,
                          price: float,
                          quantity: float,
                          capital_remaining: float) -> None:
        # Track holdings & cash
        self.capital_remaining = float(capital_remaining)
        if side == Side.BUY:
            self.position += float(quantity)
        else:
            self.position -= float(quantity)

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        # Reset best prices from full book
        if bids:
            self.best_bid = bids[0][0]
        if asks:
            self.best_ask = asks[0][0]
        m = self._mid()
        if m is not None:
            self.mid_smooth = m if self.mid_smooth is None else (self.MID_EMA_ALPHA*m + (1-self.MID_EMA_ALPHA)*self.mid_smooth)

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
        # --- 1) Update momentum ---
        pts = 0
        impact = 0.0
        if event_type == "SCORE":
            # infer points
            pts = 3 if shot_type == "THREE_POINT" else (1 if shot_type == "FREE_THROW" else 2)
            impact = float(pts) * (1.0 if home_away == "home" else -1.0)
        elif event_type in ("STEAL","BLOCK"):
            impact = 1.0 * (1.0 if home_away == "home" else -1.0)
        elif event_type == "TURNOVER":
            impact = -1.0 * (1.0 if home_away == "home" else -1.0)
        # decay-update
        self.momentum = 0.98*self.momentum + impact

        # --- 2) Build features & predict probability ---
        features = self._feature_vector(
            event_type=event_type,
            home_away=home_away,
            home_score=home_score,
            away_score=away_score,
            shot_type=shot_type,
            time_seconds=time_seconds,
            points=pts
        )
        self.last_features = features
        p_hat = self._predict_prob(features)

        # --- 3) Early-learning guard ---
        elapsed = max(0.0, 2880.0 - float(time_seconds or 0.0))
        lead_thresh = self._lead_threshold(elapsed)
        can_trade = True
        if elapsed < self.EARLY_GUARD_SECONDS:
            lead = abs(float(home_score or 0) - float(away_score or 0))
            if lead < lead_thresh:
                can_trade = False

        # --- 4) Sizing via capped Kelly ---
        if can_trade:
            target_notional = self._kelly_target_notional(p_hat)
        else:
            target_notional = 0.0

        # --- 5) Convert to target position (shares) and trade toward it ---
        m = self._mid()
        if m is None:
            if event_type == "END_GAME":
                self.reset_state()
            return

        target_shares = target_notional / max(m, 1e-6)
        # move partially toward target
        desired = self.position + self.REBALANCE_FRACTION * (target_shares - self.position)
        delta = desired - self.position

        # throttle trivial or too-large moves
        if abs(delta) < self.MIN_TRADE_QTY:
            if event_type == "END_GAME":
                self.reset_state()
            return
        qty = max(-self.MAX_TRADE_QTY, min(self.MAX_TRADE_QTY, delta))

        # Side & price
        side = Side.BUY if qty > 0 else Side.SELL
        qty = abs(qty)

        # Prefer passive: post at current best; if no book on our side, cross minimal
        if side == Side.BUY:
            px = self.best_bid if self.best_bid is not None else m
            place_limit_order(side, Ticker.TEAM_A, qty, px, ioc=False)
        else:
            px = self.best_ask if self.best_ask is not None else m
            place_limit_order(side, Ticker.TEAM_A, qty, px, ioc=False)

        # Reset state at end
        if event_type == "END_GAME":
            self.reset_state()
            return