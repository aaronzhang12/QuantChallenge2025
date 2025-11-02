# backtester.py
import json
import heapq
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import time
import random

TICK = 0.1                    # enforce price tick
MIN_PRICE, MAX_PRICE = 1.0, 99.0
WORST_CASE_BUFFER = 0.0       # require terminal equity >= this (0 is fine)

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

@dataclass
class Order:
    order_id: str
    trader_id: str
    ticker: str
    side: Side
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    ioc: bool = False
    timestamp: float = 0.0

@dataclass
class Trade:
    buy_order_id: str
    sell_order_id: str
    buy_trader_id: str
    sell_trader_id: str
    ticker: str
    quantity: int
    price: float
    timestamp: float
    aggressive_side: Side

@dataclass
class GameEvent:
    home_away: str
    home_score: int
    away_score: int
    event_type: str
    player_name: Optional[str]
    substituted_player_name: Optional[str]
    shot_type: Optional[str]
    assist_player: Optional[str]
    rebound_type: Optional[str]
    coordinate_x: Optional[float]
    coordinate_y: Optional[float]
    time_seconds: float

def _clip_price(p: float) -> float:
    # round to nearest tick and clamp
    p = round(p / TICK) * TICK
    return max(MIN_PRICE, min(MAX_PRICE, p))

class OrderBook:
    """
    Price-time priority.
    Bids: max-heap by price (store negative), Asks: min-heap by price.
    We maintain side-specific price level sizes for on_orderbook_update().
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.bids: List[Tuple[float, float, str]] = []  # (-price, ts, order_id)
        self.asks: List[Tuple[float, float, str]] = []  # (price, ts, order_id)
        self.order_map: Dict[str, Order] = {}
        self.price_levels_bid = defaultdict(int)  # price -> qty
        self.price_levels_ask = defaultdict(int)  # price -> qty

    def _add_level(self, order: Order, qty_delta: int):
        if order.side == Side.BUY:
            self.price_levels_bid[order.price] += qty_delta
            if self.price_levels_bid[order.price] <= 0:
                self.price_levels_bid.pop(order.price, None)
        else:
            self.price_levels_ask[order.price] += qty_delta
            if self.price_levels_ask[order.price] <= 0:
                self.price_levels_ask.pop(order.price, None)

    def add_order(self, order: Order) -> List[Trade]:
        order.price = _clip_price(order.price) if order.price is not None else order.price
        trades: List[Trade] = []
        if order.order_type == OrderType.MARKET:
            trades = self._execute_market_order(order)
        else:
            trades = self._add_limit_order(order)
        return trades

    def _execute_market_order(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        remaining = order.quantity

        if order.side == Side.BUY:
            # consume asks
            while remaining > 0 and self.asks:
                best_ask_price, ask_ts, ask_id = self.asks[0]
                ask = self.order_map[ask_id]
                trade_qty = min(remaining, ask.quantity)
                trades.append(Trade(
                    buy_order_id=order.order_id,
                    sell_order_id=ask.order_id,
                    buy_trader_id=order.trader_id,
                    sell_trader_id=ask.trader_id,
                    ticker=order.ticker,
                    quantity=trade_qty,
                    price=best_ask_price,
                    timestamp=order.timestamp,
                    aggressive_side=Side.BUY
                ))
                ask.quantity -= trade_qty
                remaining -= trade_qty
                self._add_level(ask, -trade_qty)
                if ask.quantity == 0:
                    heapq.heappop(self.asks)
                    del self.order_map[ask.order_id]
        else:
            # consume bids
            while remaining > 0 and self.bids:
                neg_price, bid_ts, bid_id = self.bids[0]
                bid_price = -neg_price
                bid = self.order_map[bid_id]
                trade_qty = min(remaining, bid.quantity)
                trades.append(Trade(
                    buy_order_id=bid.order_id,
                    sell_order_id=order.order_id,
                    buy_trader_id=bid.trader_id,
                    sell_trader_id=order.trader_id,
                    ticker=order.ticker,
                    quantity=trade_qty,
                    price=bid_price,
                    timestamp=order.timestamp,
                    aggressive_side=Side.SELL
                ))
                bid.quantity -= trade_qty
                remaining -= trade_qty
                self._add_level(bid, -trade_qty)
                if bid.quantity == 0:
                    heapq.heappop(self.bids)
                    del self.order_map[bid.order_id]

        return trades

    def _add_limit_order(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        remaining = order.quantity

        if order.side == Side.BUY:
            # match against asks <= order.price
            while remaining > 0 and self.asks and self.asks[0][0] <= order.price:
                best_ask_price, ask_ts, ask_id = self.asks[0]
                ask = self.order_map[ask_id]
                trade_qty = min(remaining, ask.quantity)
                trades.append(Trade(
                    buy_order_id=order.order_id,
                    sell_order_id=ask.order_id,
                    buy_trader_id=order.trader_id,
                    sell_trader_id=ask.trader_id,
                    ticker=order.ticker,
                    quantity=trade_qty,
                    price=best_ask_price,
                    timestamp=order.timestamp,
                    aggressive_side=Side.BUY
                ))
                ask.quantity -= trade_qty
                remaining -= trade_qty
                self._add_level(ask, -trade_qty)
                if ask.quantity == 0:
                    heapq.heappop(self.asks)
                    del self.order_map[ask.order_id]
            if remaining > 0 and not order.ioc:
                order.quantity = remaining
                self.order_map[order.order_id] = order
                self._add_level(order, remaining)
                heapq.heappush(self.bids, (-order.price, order.timestamp, order.order_id))
        else:
            # match against bids >= order.price
            while remaining > 0 and self.bids and -self.bids[0][0] >= order.price:
                neg_price, bid_ts, bid_id = self.bids[0]
                bid_price = -neg_price
                bid = self.order_map[bid_id]
                trade_qty = min(remaining, bid.quantity)
                trades.append(Trade(
                    buy_order_id=bid.order_id,
                    sell_order_id=order.order_id,
                    buy_trader_id=bid.trader_id,
                    sell_trader_id=order.trader_id,
                    ticker=order.ticker,
                    quantity=trade_qty,
                    price=bid_price,
                    timestamp=order.timestamp,
                    aggressive_side=Side.SELL
                ))
                bid.quantity -= trade_qty
                remaining -= trade_qty
                self._add_level(bid, -trade_qty)
                if bid.quantity == 0:
                    heapq.heappop(self.bids)
                    del self.order_map[bid.order_id]
            if remaining > 0 and not order.ioc:
                order.quantity = remaining
                self.order_map[order.order_id] = order
                self._add_level(order, remaining)
                heapq.heappush(self.asks, (order.price, order.timestamp, order.order_id))

        return trades

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.order_map:
            return False
        order = self.order_map[order_id]
        self._add_level(order, -order.quantity)
        if order.side == Side.BUY:
            self.bids = [(p, t, oid) for (p, t, oid) in self.bids if oid != order_id]
            heapq.heapify(self.bids)
        else:
            self.asks = [(p, t, oid) for (p, t, oid) in self.asks if oid != order_id]
            heapq.heapify(self.asks)
        del self.order_map[order_id]
        return True

    def get_best_bid(self) -> Optional[float]:
        return (-self.bids[0][0]) if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    def get_level_quantity(self, price: float, side: Side) -> int:
        if side == Side.BUY:
            return self.price_levels_bid.get(price, 0)
        return self.price_levels_ask.get(price, 0)

    def decay_quotes(self, cancel_frac: float = 0.15):
        """
        Randomly cancel a fraction of resting quotes to mimic quote churn.
        """
        # bids
        bid_ids = [oid for _, _, oid in self.bids]
        random.shuffle(bid_ids)
        n_cancel = int(len(bid_ids) * cancel_frac)
        for oid in bid_ids[:n_cancel]:
            self.cancel_order(oid)
        # asks
        ask_ids = [oid for _, _, oid in self.asks]
        random.shuffle(ask_ids)
        n_cancel = int(len(ask_ids) * cancel_frac)
        for oid in ask_ids[:n_cancel]:
            self.cancel_order(oid)

class TradingAlgorithm:
    def __init__(self, trader_id: str, capital: float = 100000.0):
        self.trader_id = trader_id
        self.initial_capital = capital
        self.capital_remaining = capital
        self.positions = defaultdict(int)  # ticker -> quantity
        self.pnl = 0.0

    # UPDATED SIGNATURE: include side for clarity (BID/ASK)
    def on_orderbook_update(self, ticker: str, side: Side, price: float, quantity: int):
        pass

    def on_trade_update(self, trade: Trade):
        pass

    def on_account_update(self, order_id: str, quantity: int, price: float,
                         capital_remaining: float):
        pass

    def on_game_event_update(self, event: GameEvent):
        pass

    def place_market_order(self, ticker: str, side: Side, quantity: int) -> str:
        pass

    def place_limit_order(self, ticker: str, side: Side, quantity: int,
                         price: float, ioc: bool = False) -> str:
        pass

    def cancel_order(self, ticker: str, order_id: str) -> bool:
        pass

class BacktesterEngine:
    def __init__(self):
        self.orderbooks: Dict[str, OrderBook] = {}
        self.algorithms: Dict[str, TradingAlgorithm] = {}
        self.next_order_id = 1
        self.current_time = 0.0
        self.game_events: List[GameEvent] = []
        self.trades_history: List[Trade] = []
        self.last_trade_price: Dict[str, float] = {}

    def add_ticker(self, ticker: str):
        if ticker not in self.orderbooks:
            self.orderbooks[ticker] = OrderBook(ticker)

    def add_algorithm(self, algorithm: TradingAlgorithm):
        self.algorithms[algorithm.trader_id] = algorithm

        # bind trading fns
        def place_market_order(ticker, side, quantity):
            return self._place_market_order(algorithm.trader_id, ticker, side, quantity)
        def place_limit_order(ticker, side, quantity, price, ioc=False):
            return self._place_limit_order(algorithm.trader_id, ticker, side, quantity, price, ioc)
        def cancel_order(ticker, order_id):
            return self._cancel_order(ticker, order_id)

        algorithm.place_market_order = place_market_order
        algorithm.place_limit_order = place_limit_order
        algorithm.cancel_order = cancel_order

    # -------- Risk & capital checks (conservative) --------
    def _would_violate_worst_case(self, algo: TradingAlgorithm, ticker: str,
                                  side: Side, qty: int, price: float) -> bool:
        """
        Simulate cash/position after the proposed trade, then require terminal
        equity >= WORST_CASE_BUFFER under the adverse settlement:
          - If final Q' >= 0 (net long): adverse S=0
          - If final Q' < 0  (net short): adverse S=100
        """
        Q = algo.positions[ticker]
        C = algo.capital_remaining

        if side == Side.BUY:
            Qp = Q + qty
            Cp = C - price * qty
        else:
            Qp = Q - qty
            Cp = C + price * qty

        adverse_S = 0.0 if Qp >= 0 else 100.0
        terminal_equity = Cp + Qp * adverse_S
        return terminal_equity < WORST_CASE_BUFFER

    def _ensure_cash(self, algo: TradingAlgorithm, side: Side, qty: int, price: float) -> int:
        """
        Clip BUY size so cost <= cash. SELLs (shorts) are not cash-limited
        here, but still go through worst-case equity check.
        """
        if side == Side.BUY and price > 0:
            affordable = int(algo.capital_remaining // price)
            return max(0, min(qty, affordable))
        return qty

    # -------- Order entry with checks --------
    def _place_market_order(self, trader_id: str, ticker: str, side: Side, quantity: int) -> str:
        if ticker not in self.orderbooks or quantity <= 0:
            return ""
        ob = self.orderbooks[ticker]
        best_ask = ob.get_best_ask()
        best_bid = ob.get_best_bid()
        est_price = (best_ask if side == Side.BUY else best_bid)
        if est_price is None:
            # no liquidity on opposite side
            return ""

        algo = self.algorithms[trader_id]
        qty = self._ensure_cash(algo, side, quantity, est_price)
        if qty <= 0:
            return ""

        # Worst-case check at current best price (conservative)
        if self._would_violate_worst_case(algo, ticker, side, qty, est_price):
            return ""

        order_id = f"order_{self.next_order_id}"
        self.next_order_id += 1
        order = Order(
            order_id=order_id, trader_id=trader_id, ticker=ticker,
            side=side, order_type=OrderType.MARKET, quantity=qty,
            timestamp=self.current_time
        )
        trades = self.orderbooks[ticker].add_order(order)
        self._process_trades(trades)
        return order_id

    def _place_limit_order(self, trader_id: str, ticker: str, side: Side,
                           quantity: int, price: float, ioc: bool = False) -> str:
        if ticker not in self.orderbooks or quantity <= 0:
            return ""
        price = _clip_price(price)
        algo = self.algorithms[trader_id]

        qty = self._ensure_cash(algo, side, quantity, price)
        if qty <= 0:
            return ""

        # Worst-case check at limit price (conservative)
        if self._would_violate_worst_case(algo, ticker, side, qty, price):
            return ""

        order_id = f"order_{self.next_order_id}"
        self.next_order_id += 1
        order = Order(
            order_id=order_id, trader_id=trader_id, ticker=ticker,
            side=side, order_type=OrderType.LIMIT, quantity=qty,
            price=price, ioc=ioc, timestamp=self.current_time
        )
        trades = self.orderbooks[ticker].add_order(order)
        self._process_trades(trades)
        return order_id

    def _cancel_order(self, ticker: str, order_id: str) -> bool:
        if ticker not in self.orderbooks:
            return False
        return self.orderbooks[ticker].cancel_order(order_id)

    # -------- Trade settlement / notifications --------
    def _process_trades(self, trades: List[Trade]):
        for trade in trades:
            self.trades_history.append(trade)
            self.last_trade_price[trade.ticker] = trade.price

            # Look up counterparties; external (e.g., SEED) may be absent
            buy_algo = self.algorithms.get(trade.buy_trader_id, None)
            sell_algo = self.algorithms.get(trade.sell_trader_id, None)

            # Update buyer state if it's one of our registered algos
            if buy_algo is not None:
                buy_algo.positions[trade.ticker] += trade.quantity
                buy_algo.capital_remaining -= trade.price * trade.quantity
                buy_algo.on_account_update(
                    trade.buy_order_id, trade.quantity, trade.price, buy_algo.capital_remaining
                )

            # Update seller state if it's one of our registered algos
            if sell_algo is not None:
                sell_algo.positions[trade.ticker] -= trade.quantity
                sell_algo.capital_remaining += trade.price * trade.quantity
                sell_algo.on_account_update(
                    trade.sell_order_id, -trade.quantity, trade.price, sell_algo.capital_remaining
                )

            # Broadcast trade to all registered algos (external parties don't need notifications)
            for algo in self.algorithms.values():
                algo.on_trade_update(trade)


    def _notify_orderbook_updates(self, ticker: str):
        ob = self.orderbooks[ticker]

        # send best levels first
        best_bid = ob.get_best_bid()
        best_ask = ob.get_best_ask()
        if best_bid is not None:
            for algo in self.algorithms.values():
                qty = ob.get_level_quantity(best_bid, Side.BUY)
                if qty > 0:
                    algo.on_orderbook_update(ticker, Side.BUY, best_bid, qty)
        if best_ask is not None:
            for algo in self.algorithms.values():
                qty = ob.get_level_quantity(best_ask, Side.SELL)
                if qty > 0:
                    algo.on_orderbook_update(ticker, Side.SELL, best_ask, qty)

        # Optionally broadcast more depth (commented to reduce spam)
        # for price, qty in ob.price_levels_bid.items():
        #     for algo in self.algorithms.values():
        #         algo.on_orderbook_update(ticker, Side.BUY, price, qty)
        # for price, qty in ob.price_levels_ask.items():
        #     for algo in self.algorithms.values():
        #         algo.on_orderbook_update(ticker, Side.SELL, price, qty)

    def load_game_events(self, json_file: str):
        with open(json_file, 'r') as f:
            events_data = json.load(f)
        self.game_events = [GameEvent(**ed) for ed in events_data]
        self.game_events.sort(key=lambda x: x.time_seconds, reverse=True)

    def run_backtest(self, ticker: str = "BASKETBALL"):
        self.add_ticker(ticker)

        for trader_id, algo in list(self.algorithms.items()):
            preserved = {
                'logger': getattr(algo, 'logger', None),
            }
            algo.__init__(algo.trader_id, algo.initial_capital)
            # restore preserved attrs
            for k, v in preserved.items():
                if v is not None:
                    setattr(algo, k, v)

        print(f"Starting backtest with {len(self.algorithms)} algorithms")
        print(f"Processing {len(self.game_events)} game events")

        # Seed standing book with a tiny 2-tick spread around 50 to start
        ob = self.orderbooks[ticker]
        seed_ts = 1e9
        def seed_limit(side: Side, price: float, qty: int):
            oid = f"seed_{side.value}_{price}"
            order = Order(
                order_id=oid, trader_id="SEED", ticker=ticker, side=side,
                order_type=OrderType.LIMIT, quantity=qty, price=_clip_price(price),
                ioc=False, timestamp=seed_ts
            )
            ob.order_map[oid] = order
            ob._add_level(order, qty)
            if side == Side.BUY:
                heapq.heappush(ob.bids, (-order.price, order.timestamp, oid))
            else:
                heapq.heappush(ob.asks, (order.price, order.timestamp, oid))
        seed_limit(Side.BUY, 49.9, 200)
        seed_limit(Side.SELL, 50.1, 200)

        # process events
        for i, event in enumerate(self.game_events):
            self.current_time = event.time_seconds

            # Notify algos of the game event
            for algo in self.algorithms.values():
                algo.on_game_event_update(event)

            # Random quote decay to mimic churn
            if random.random() < 0.35:
                ob.decay_quotes(cancel_frac=random.uniform(0.08, 0.22))

            # After algos reacted, notify best bid/ask
            self._notify_orderbook_updates(ticker)

            if i % 100 == 0:
                print(f"Processed {i}/{len(self.game_events)} events, time: {event.time_seconds:.1f}s")

        # Final settlement
        self._settle_positions(ticker)

        # Results
        results = self._calculate_results()
        return results

    def _settle_positions(self, ticker: str):
        final_event = self.game_events[-1]
        home_wins = final_event.home_score > final_event.away_score
        payout = 100.0 if home_wins else 0.0
        print(f"Final score: Home {final_event.home_score} - Away {final_event.away_score}")
        print(f"Home team {'WINS' if home_wins else 'LOSES'}, payout: ${payout}/share")
        for algo in self.algorithms.values():
            pos = algo.positions[ticker]
            algo.capital_remaining += pos * payout
            algo.pnl = algo.capital_remaining - algo.initial_capital
            print(f"{algo.trader_id}: Position={pos}, Final PnL=${algo.pnl:.2f}")

    def _calculate_results(self) -> Dict[str, Any]:
        sorted_algos = sorted(self.algorithms.values(), key=lambda x: x.pnl, reverse=True)
        results = {
            'rankings': [],
            'final_scores': {},
            'total_trades': len(self.trades_history),
            'algorithms': len(self.algorithms)
        }
        for rank, algo in enumerate(sorted_algos, 1):
            results['rankings'].append({
                'rank': rank,
                'trader_id': algo.trader_id,
                'pnl': algo.pnl,
                'final_capital': algo.capital_remaining,
                'return_pct': (algo.pnl / algo.initial_capital) * 100
            })
            results['final_scores'][algo.trader_id] = algo.pnl
        return results
