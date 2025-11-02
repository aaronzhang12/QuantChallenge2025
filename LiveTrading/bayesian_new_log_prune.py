import math
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

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
    A Bayesian strategy with dynamic risk management. It reduces its Kelly scaling
    as it accumulates realized profits, and takes profit on positions earlier.
    """
    # ======= Risk/behavior knobs =======
    BASE_KELLY_SCALE = 0.05      # The baseline Kelly scaling factor.
    MIN_KELLY_SCALE = 0.01       # The minimum Kelly scaling, to avoid stopping trading.
    PROFIT_SCALING_FACTOR = 0.001 # How much profits reduce the Kelly scale.
    MAX_POSITION_SHARES = 80.0
    MAX_ORDER_SHARES = 10.0
    TRADE_COOLDOWN_MS = 5000
    EDGE_ENTRY_EARLY  = 0.080
    EDGE_ENTRY_MID    = 0.070
    EDGE_ENTRY_LATE   = 0.060
    EDGE_ADD          = 0.100
    MAX_CROSS_SPREAD = 1.50
    PASSIVE_NUDGE     = 0.10
    TAKE_PROFIT    = 1.75      # REDUCED: Take profits earlier.
    SOFT_STOP_LOSS = 3.25
    MIN_PRICE = 1.0
    MAX_PRICE = 99.0
    TICK = 0.1

    # ======= Model hyperparameters =======
    NOISE_VAR: float      = 9.0
    PRIOR_VAR: float      = 400.0

    # ======= Feature schema =======
    _EVENTS = ("SCORE","STEAL","TURNOVER","BLOCK","FOUL","MISSED", "SUBSTITUTION","TIMEOUT","END_PERIOD","END_GAME","REBOUND","JUMP_BALL")
    _SHOTS  = ("THREE_POINT","TWO_POINT","DUNK","LAYUP","FREE_THROW")
    _D = 1 + 2 + 4 + 3 + 2 + len(_EVENTS) + len(_SHOTS) + 2 + 5

    def reset_state(self) -> None:
        self.mu: List[float] = [0.0] * self._D
        self.mu[0] = 50.0; self.mu[5] = 50.0; self.mu[8] = 5.0; self.mu[9] = 2.5
        self.Sigma: List[List[float]] = [[0.0]*self._D for _ in range(self._D)]
        for i in range(self._D): self.Sigma[i][i] = self.PRIOR_VAR
        self.best_bid: Optional[float] = None; self.best_ask: Optional[float] = None
        self.last_trade: Optional[float] = None; self.mid_price: float = 50.0
        self.position: float = 0.0; self.cash: float = 100000.0
        self.avg_entry: Optional[float] = None; self.last_order_ms: int = 0
        self._last_x: Optional[List[float]] = None; self._score_diffs: List[float] = []
        self._events_hist: List[str] = []; self._home_aways: List[str] = []
        self._player_scores_hist: List[Tuple[str, int, str, float]] = []
        self._player_shot_attempts_hist: List[Tuple[str, str, str, float]] = []
        self.game_start_time_seconds: Optional[float] = None
        self.learn_until: Optional[float] = None
        self.realized_pnl: float = 0.0 # NEW: Track realized PnL

    def __init__(self) -> None:
        self.reset_state()

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float: return sum(ai*bi for ai,bi in zip(a,b))
    @staticmethod
    def _mat_vec(M: List[List[float]], v: List[float]) -> List[float]: return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
    @staticmethod
    def _outer(u: List[float], v: List[float]) -> List[List[float]]: return [[ui*vj for vj in v] for ui in u]
    @staticmethod
    def _mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        m, n = len(A), len(A[0]) if A else 0
        return [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]

    def _bayes_update(self, x: List[float], y: float, lr: float = 0.9) -> None:
        Sx = self._mat_vec(self.Sigma, x)
        denom = self.NOISE_VAR + self._dot(x, Sx)
        if denom <= 1e-9: return
        K = [lr * (v/denom) for v in Sx]
        resid = y - self._dot(self.mu, x)
        for i in range(self._D): self.mu[i] += K[i] * resid
        self.Sigma = self._mat_sub(self.Sigma, self._outer(K, Sx))

    def _features(self, event_type: str, home_away: str, home_score: int, away_score: int,
                  player_name: Optional[str], shot_type: Optional[str],
                  time_seconds: Optional[float]) -> List[float]:
        t = float(time_seconds or 0.0)
        max_game_time = self.game_start_time_seconds or 2880.0
        t = max(0.0, min(t, max_game_time))
        t_norm, t_left = t / max_game_time, 1.0 - (t / max_game_time)
        sd, ad = float(home_score - away_score), abs(float(home_score - away_score))
        sd_t, ad_t = sd * t_left, ad * t_left
        self._score_diffs.append(sd); self._events_hist.append(event_type or ""); self._home_aways.append(home_away or "")
        if len(self._score_diffs) > 120: self._score_diffs.pop(0); self._events_hist.pop(0); self._home_aways.pop(0)
        sd_chg_1 = self._score_diffs[-1] - self._score_diffs[-2] if len(self._score_diffs) > 1 else 0.0
        sd_mom_10 = self._score_diffs[-1] - self._score_diffs[-11] if len(self._score_diffs) > 10 else 0.0
        sd_mom_30 = self._score_diffs[-1] - self._score_diffs[-31] if len(self._score_diffs) > 30 else 0.0
        run_len = 0.0
        if self._events_hist and self._events_hist[-1] == "SCORE":
            team = self._home_aways[-1]
            for i in range(len(self._events_hist) - 1, -1, -1):
                if self._events_hist[i] == "SCORE" and self._home_aways[i] == team: run_len += 1
                else: break
        pace_60 = float(sum(1 for e in self._events_hist[-60:] if e == "SCORE"))
        ev_oh = [1.0 if event_type == e else 0.0 for e in self._EVENTS]
        sh_oh = [1.0 if shot_type == s else 0.0 for s in self._SHOTS]
        spread, mid_dev = 0.0, 0.0
        if self.best_bid is not None and self.best_ask is not None:
            mid = 0.5 * (self.best_bid + self.best_ask)
            spread = abs(self.best_ask - self.best_bid)
            mid_dev = (self.last_trade or mid) - mid
        if event_type == "SCORE" and player_name:
            pts = {"FREE_THROW": 1, "TWO_POINT": 2, "DUNK": 2, "LAYUP": 2, "THREE_POINT": 3}.get(shot_type, 0)
            self._player_scores_hist.append((player_name, pts, home_away, t))
        if shot_type in self._SHOTS and player_name:
            self._player_shot_attempts_hist.append((player_name, shot_type, home_away, t))
        window_start_time = t - 60.0
        self._player_scores_hist = [item for item in self._player_scores_hist if item[3] >= window_start_time]
        self._player_shot_attempts_hist = [item for item in self._player_shot_attempts_hist if item[3] >= window_start_time]
        player_score_impact = sum(s for p, s, _, _ in self._player_scores_hist if p == player_name) if player_name else 0.0
        player_shot_attempt_impact = sum(1 for p, _, _, _ in self._player_shot_attempts_hist if p == player_name) if player_name else 0.0
        home_team_recent_pts = sum(s for _, s, ha, _ in self._player_scores_hist if ha == "home")
        away_team_recent_pts = sum(s for _, s, ha, _ in self._player_scores_hist if ha == "away")
        x = ([1.0] + [t_norm, t_left] + [sd, ad, sd_t, ad_t] + [sd_chg_1, sd_mom_10, sd_mom_30] +
             [run_len, pace_60] + ev_oh + sh_oh + [spread, mid_dev] +
             [player_score_impact, player_shot_attempt_impact, 1.0 if event_type == "SUBSTITUTION" else 0.0,
              home_team_recent_pts, away_team_recent_pts])
        return x + [0.0] * (self._D - len(x))

    def _clip(self, x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
    def _clip_price(self, p: float) -> float: return round(self._clip(p, self.MIN_PRICE, self.MAX_PRICE) / self.TICK) * self.TICK
    def _now_ms(self) -> int: return int(time.time() * 1000)
    def _update_mid(self):
        if self.best_bid is not None and self.best_ask is not None: self.mid_price = (self.best_bid + self.best_ask) / 2.0
        elif self.last_trade is not None: self.mid_price = self.last_trade
    def _relative_phase_edge(self, time_seconds: Optional[float]) -> float:
        if self.game_start_time_seconds is None or time_seconds is None: return self.EDGE_ENTRY_MID
        frac = time_seconds / max(1.0, self.game_start_time_seconds)
        if frac > 0.70: return self.EDGE_ENTRY_EARLY
        elif frac > 0.40: return self.EDGE_ENTRY_MID
        else: return self.EDGE_ENTRY_LATE
    def _update_avg_entry_after_fill(self, side: Side, qty: float, price: float):
        pos0 = self.position
        if side == Side.BUY:
            if pos0 >= 0: self.avg_entry = price if self.avg_entry is None else (self.avg_entry * pos0 + price * qty) / max(1.0, pos0 + qty)
            elif pos0 + qty >= 0: self.avg_entry = price
        else:
            if pos0 <= 0: self.avg_entry = price if self.avg_entry is None else (self.avg_entry * (-pos0) + price * qty) / max(1.0, -pos0 + qty)
            elif pos0 - qty <= 0: self.avg_entry = price
            
    def _dynamic_kelly_scale(self) -> float:
        """ Adjusts Kelly scaling based on realized PnL. """
        if self.realized_pnl > 0:
            scale = self.BASE_KELLY_SCALE / (1 + self.realized_pnl * self.PROFIT_SCALING_FACTOR)
            return max(self.MIN_KELLY_SCALE, scale)
        return self.BASE_KELLY_SCALE

    def _maybe_take_profit_or_stop(self) -> bool:
        if self.position == 0 or self.mid_price is None or self.avg_entry is None: return False
        pnl_ps = (self.mid_price - self.avg_entry) if self.position > 0 else (self.avg_entry - self.mid_price)
        if pnl_ps >= self.TAKE_PROFIT or pnl_ps <= -self.SOFT_STOP_LOSS:
            qty = min(abs(self.position), self.MAX_ORDER_SHARES)
            place_market_order(Side.SELL if self.position > 0 else Side.BUY, Ticker.TEAM_A, qty)
            return True
        return False
        
    def _kelly_target_shares(self, p_hat: float, s_prob: float, price_dollars: float, cash: float) -> float:
        p, s = self._clip(p_hat, 1e-4, 1-1e-4), self._clip(s_prob, 1e-4, 1-1e-4)
        edge = p - s
        KELLY_SCALE = self._dynamic_kelly_scale() # Use the dynamic scale
        f = (edge / (1.0 - s) if edge > 0 else edge / s) * KELLY_SCALE
        return self._clip((f * cash) / max(1.0, price_dollars), -self.MAX_POSITION_SHARES, self.MAX_POSITION_SHARES)

    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != Ticker.TEAM_A: return
        self.last_trade = float(price)
        self._update_mid()
        if self._last_x is not None: self._bayes_update(self._last_x, self.last_trade)

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != Ticker.TEAM_A: return
        if side == Side.BUY: self.best_bid = float(price) if self.best_bid is None or price > self.best_bid else self.best_bid
        else: self.best_ask = float(price) if self.best_ask is None or price < self.best_ask else self.best_ask
        self._update_mid()

    def on_account_update(self, ticker: Ticker, side: Side, price: float, quantity: float, capital_remaining: float) -> None:
        if ticker != Ticker.TEAM_A: return

        # --- Realized PnL Calculation ---
        pos_before = self.position
        is_closing_trade = (pos_before > 0 and side == Side.SELL) or (pos_before < 0 and side == Side.BUY)
        if is_closing_trade and self.avg_entry is not None:
            closed_qty = min(abs(pos_before), quantity)
            pnl_per_share = (price - self.avg_entry) if pos_before > 0 else (self.avg_entry - price)
            self.realized_pnl += pnl_per_share * closed_qty
        
        self._update_avg_entry_after_fill(side, quantity, price)
        self.position += quantity if side == Side.BUY else -quantity
        self.cash = capital_remaining
        self.position = self._clip(self.position, -self.MAX_POSITION_SHARES, self.MAX_POSITION_SHARES)

    def on_game_event_update(self, event_type: str, home_away: str, home_score: int, away_score: int,
                             player_name: Optional[str], substituted_player_name: Optional[str],
                             shot_type: Optional[str], assist_player: Optional[str],
                             rebound_type: Optional[str], coordinate_x: Optional[float],
                             coordinate_y: Optional[float], time_seconds: Optional[float]) -> None:
        if event_type == "END_GAME": self.reset_state(); return
        if self.game_start_time_seconds is None and time_seconds is not None:
            self.game_start_time_seconds = float(time_seconds)
            self.learn_until = max(0.0, self.game_start_time_seconds - 240.0)
        t = float(time_seconds or 0.0)
        x = self._features(event_type, home_away, home_score, away_score, player_name, shot_type, t)
        self._last_x = x
        fair_value = self._clip_price(self._dot(self.mu, x))
        if time_seconds is None or (self.learn_until is not None and time_seconds > self.learn_until): return
        if self._maybe_take_profit_or_stop(): self.last_order_ms = self._now_ms(); return
        self._update_mid()
        mid = self._clip(self.mid_price, self.MIN_PRICE, self.MAX_PRICE)
        p_mkt = self._clip(mid / 100.0, 1e-3, 1 - 1e-3)
        p_hat = self._clip(fair_value / 100.0, 1e-3, 1 - 1e-3)
        edge = p_hat - p_mkt
        thr = self._relative_phase_edge(time_seconds)
        if self._now_ms() - self.last_order_ms < self.TRADE_COOLDOWN_MS or abs(edge) < thr: return
        target = self._kelly_target_shares(p_hat, p_mkt, mid, self.cash)
        if abs(edge) < self.EDGE_ADD and abs(target) > abs(self.position):
            target = self.position + (target - self.position) * 0.5
        delta = target - self.position
        if abs(delta) < 1.0: return
        qty = min(self.MAX_ORDER_SHARES, abs(delta))
        side = Side.BUY if delta > 0 else Side.SELL
        spread = abs(self.best_ask - self.best_bid) if self.best_bid and self.best_ask else float('inf')
        if spread <= self.MAX_CROSS_SPREAD:
            place_market_order(side, Ticker.TEAM_A, qty)
            self.last_order_ms = self._now_ms()
        else:
            px = self.best_bid + self.PASSIVE_NUDGE if side == Side.BUY and self.best_bid else (mid - self.PASSIVE_NUDGE if side == Side.BUY else (self.best_ask - self.PASSIVE_NUDGE if self.best_ask else mid + self.PASSIVE_NUDGE))
            place_limit_order(side, Ticker.TEAM_A, qty, self._clip_price(px), ioc=False)

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        pass