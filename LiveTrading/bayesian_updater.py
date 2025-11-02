# bayesian_strategy.py
import math
import numpy as np
from typing import Optional, List
from backtester import TradingAlgorithm, Side, GameEvent, Trade

class BayesianRegressionStrategy(TradingAlgorithm):
    """
    Bayesian regression baseline that:
      - predicts market price from game state features
      - trades only when the edge is meaningful (more conservative early)
      - updates its posterior online from observed trade prints
    """

    def __init__(self, trader_id: str, capital: float = 100000.0,
                 model_path: str = "models/bayes_market_model.npz",
                 min_elapsed_sec: int = 600,             # don't ‘decide’ too early
                 base_threshold: float = 1.8,            # edge threshold in $ for crossing
                 max_position: int = 250,
                 ioc_limits: bool = True):
        super().__init__(trader_id, capital)
        self.model_path = model_path
        self.min_elapsed_sec = min_elapsed_sec
        self.base_threshold = base_threshold
        self.max_position = max_position
        self.ioc_limits = ioc_limits

        self._load_model()
        self._last_trade_price: Optional[float] = None
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None

        # small buffer for momentum/pace features
        self._score_diffs: List[float] = []
        self._events: List[str] = []
        self._home_aways: List[str] = []
        self._last_prices: List[float] = []

    # ---------- model io ----------
    def _load_model(self):
        m = np.load(self.model_path, allow_pickle=True)
        self.mu = m["mu"]                  # (d,)
        self.Sigma = m["Sigma"]            # (d,d)
        self.sigma2 = float(m["sigma2"][0])  # standardized noise variance (1.0)
        self.X_mean = m["X_mean"]          # (d,)
        self.X_std  = m["X_std"]           # (d,)
        self.y_mean = float(m["y_mean"][0])
        self.y_std  = float(m["y_std"][0])
        self.feature_names = m["feature_names"].tolist()

    # ---------- features ----------
    def _feat(self, ev: GameEvent) -> np.ndarray:
        # Base
        t = float(ev.time_seconds)
        t_norm = t/2880.0
        t_left = 1.0 - t_norm
        score_diff = float(ev.home_score - ev.away_score)
        abs_diff = abs(score_diff)
        home_sign = 1.0 if ev.home_away=="home" else (-1.0 if ev.home_away=="away" else 0.0)

        # Momentum-esque buffers (by events, not exact seconds)
        self._score_diffs.append(score_diff)
        self._events.append(ev.event_type or "")
        self._home_aways.append(ev.home_away or "")
        if len(self._score_diffs) > 120: self._score_diffs.pop(0)
        if len(self._events) > 120: self._events.pop(0)
        if len(self._home_aways) > 120: self._home_aways.pop(0)

        # score momentum
        sd = self._score_diffs
        sd_chg_1 = 0.0 if len(sd)<2 else (sd[-1] - sd[-2])
        sd_mom_10 = 0.0 if len(sd)<11 else (sd[-10] - sd[-1])
        sd_mom_30 = 0.0 if len(sd)<31 else (sd[-30] - sd[-1])

        # run length (consecutive same-team SCORE)
        run_len = 0.0
        if self._events:
            i = len(self._events)-1
            if self._events[i] == "SCORE":
                team = self._home_aways[i]
                run_len = 1.0
                j = i-1
                while j>=0 and self._events[j]=="SCORE" and self._home_aways[j]==team:
                    run_len += 1.0
                    j -= 1

        # pace: SCOREs in last 60 events
        recent_evs = self._events[-60:]
        pace_60 = sum(1.0 for e in recent_evs if e=="SCORE")

        # event/shot one-hots (same set as trainer)
        evs = ["SCORE","STEAL","TURNOVER","BLOCK","FOUL","MISSED","SUBSTITUTION","TIMEOUT","END_PERIOD","END_GAME"]
        shots = ["THREE_POINT","TWO_POINT","DUNK","LAYUP","FREE_THROW"]
        ev_onehots = [1.0 if (ev.event_type==e) else 0.0 for e in evs]
        shot_onehots = [1.0 if (ev.shot_type==s) else 0.0 for s in shots]

        # simple microstructure hints from the book if available
        spread = 0.0
        mid_dev = 0.0
        if self._best_bid is not None and self._best_ask is not None:
            spread = abs(self._best_ask - self._best_bid)
            mid0 = (self._best_bid + self._best_ask)/2.0
            last = self._last_trade_price if self._last_trade_price is not None else mid0
            mid_dev = (last - mid0)

        x = []
        # NOTE: order must match the trainer’s feature order
        x += [t_norm, t_left]
        x += [score_diff, abs_diff, score_diff*t_left, abs_diff*t_left, home_sign]
        x += [sd_chg_1, sd_mom_10, sd_mom_30, run_len, float(pace_60)]
        x += ev_onehots
        x += shot_onehots

        # ensure alignment to the saved model’s feature_names
        x = np.array(x, dtype=float)
        # standardize
        x_std = (x - self.X_mean) / self.X_std
        return x_std

    # ---------- price prediction ----------
    def _predict_price(self, x_std: np.ndarray) -> float:
        y_std_hat = float(x_std @ self.mu)
        return self.y_mean + self.y_std * y_std_hat

    # ---------- online BLR update ----------
    def _bayes_update(self, x_std: np.ndarray, y_observed: float, lr: float = 1.0):
        """
        Update posterior with observation y (de-mean/scale to model space).
        lr < 1.0 acts like a gentle forgetting/robustness factor.
        """
        y_std = (y_observed - self.y_mean) / self.y_std
        Sx = self.Sigma @ x_std
        denom = self.sigma2 + float(x_std.T @ Sx)
        if denom <= 0:
            return
        K = (Sx / denom) * lr                            # (d,)
        residual = y_std - float(x_std @ self.mu)
        self.mu = self.mu + K * residual                 # (d,)
        self.Sigma = self.Sigma - np.outer(K, Sx)        # (d,d)

    # ---------- backtester hooks ----------
    def on_orderbook_update(self, ticker: str, side: Side, price: float, quantity: int):
        if ticker != "BASKETBALL" or quantity <= 0: return
        if side == Side.BUY:
            self._best_bid = price if (self._best_bid is None or price > self._best_bid) else self._best_bid
        else:
            self._best_ask = price if (self._best_ask is None or price < self._best_ask) else self._best_ask

    def on_trade_update(self, trade: Trade):
        if trade.ticker != "BASKETBALL": return
        # observe market price, use as label for online update with last x_std (if we have one)
        self._last_trade_price = trade.price
        if hasattr(self, "_last_x_std"):
            self._bayes_update(self._last_x_std, trade.price, lr=0.85)  # mild robustness

    def on_game_event_update(self, event: GameEvent):
        # Build features & predict
        x_std = self._feat(event)
        self._last_x_std = x_std  # store for online update when a trade arrives
        pred = self._predict_price(x_std)

        # Market ref
        if self._best_bid is not None and self._best_ask is not None:
            market_px = (self._best_bid + self._best_ask)/2.0
        elif self._last_trade_price is not None:
            market_px = self._last_trade_price
        else:
            market_px = 50.0

        # Conservative early: require elapsed >= min_elapsed_sec
        elapsed = 2880.0 - float(event.time_seconds)
        if elapsed < self.min_elapsed_sec:
            # passive: tiny skew around prediction to collect info, not risk
            size = 25
            if self.positions["BASKETBALL"] < self.max_position:
                self.place_limit_order("BASKETBALL", Side.BUY, size, max(1.0, pred - 1.0), ioc=self.ioc_limits)
            if self.positions["BASKETBALL"] > -self.max_position:
                self.place_limit_order("BASKETBALL", Side.SELL, size, min(99.0, pred + 1.0), ioc=self.ioc_limits)
            return

        # Edge and time-based threshold (stricter early, looser late)
        t_norm = float(event.time_seconds) / 2880.0
        threshold = self.base_threshold * (0.6 + 0.4 * t_norm)  # >= base_threshold early, ~0.6*base late
        edge = pred - market_px

        # Desired position from edge (soft Kelly-ish)
        cur = self.positions["BASKETBALL"]
        conf = min(1.0, 1.0 / (1.0 + float(x_std @ self.Sigma @ x_std)))  # larger when posterior variance small
        target = np.clip((edge / 6.0) * conf * self.max_position, -self.max_position, self.max_position)
        delta = int(target - cur)

        if abs(edge) < threshold or abs(delta) < 10:
            # provide liquidity around pred if flat-ish
            if abs(cur) < self.max_position * 0.5:
                bid = max(1.0, pred - (1.2 + 0.8*(1.0-t_norm)))
                ask = min(99.0, pred + (1.2 + 0.8*(1.0-t_norm)))
                if self.capital_remaining > bid * 40:
                    self.place_limit_order("BASKETBALL", Side.BUY, 40, bid, ioc=self.ioc_limits)
                self.place_limit_order("BASKETBALL", Side.SELL, 40, ask, ioc=self.ioc_limits)
            return

        # If we have a real edge, move towards target
        side = Side.BUY if edge > 0 else Side.SELL
        qty = min(max(10, abs(delta)), 80)  # clip trade size
        urgency = (abs(edge) - threshold) / max(1.0, threshold) + (1.0 - t_norm) * 0.5

        if urgency > 0.8:
            self.place_market_order("BASKETBALL", side, qty)
        else:
            # skewed limit towards pred to avoid crossing when not urgent
            px = pred - 0.3 if side == Side.BUY else pred + 0.3
            self.place_limit_order("BASKETBALL", side, qty, float(np.clip(px, 1.0, 99.0)), ioc=self.ioc_limits)