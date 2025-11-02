from enum import Enum
from typing import Optional, Dict, Set

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    return True

class Strategy:

    def __init__(self) -> None:
        self.PROFIT_TARGET = 40.0
        self.STOP_LOSS = 15
        self.TRADE_QUANTITY = 5.0
        self.RATING_CHANGE_THRESHOLD = 5.0
        self.reset_state()

    def reset_state(self) -> None:
        print("Resetting strategy state for a new game.")
        self.player_ratings: Dict[str, float] = {}
        self.home_players_on_court: Set[str] = set()
        self.away_players_on_court: Set[str] = set()
        self.last_home_score = 0
        self.last_away_score = 0
        self.mid_price: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.current_position: float = 0.0
        self.entry_price: float = 0.0
        self.exit_order_id: Optional[int] = None

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        pass

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        pass

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        if bids and asks:
            self.best_bid = bids[0][0]
            self.best_ask = asks[0][0]
            self.mid_price = (self.best_bid + self.best_ask) / 2.0

            if self.current_position > 0:
                if self.best_bid <= self.entry_price - self.STOP_LOSS:
                    print(f"Stop-loss triggered for LONG position. Best bid at {self.best_bid}.")
                    self._close_position_at_market()
            elif self.current_position < 0:
                if self.best_ask >= self.entry_price + self.STOP_LOSS:
                    print(f"Stop-loss triggered for SHORT position. Best ask at {self.best_ask}.")
                    self._close_position_at_market()

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        print(f"Account Update: Filled {side.name} order for {quantity} at {price}")

        if side == Side.BUY:
            total_cost_old = self.entry_price * self.current_position
            total_cost_new = price * quantity
            self.current_position += quantity
            self.entry_price = (total_cost_old + total_cost_new) / self.current_position
        else:
            total_cost_old = self.entry_price * self.current_position
            total_cost_new = price * quantity
            self.current_position -= quantity
            if abs(self.current_position) > 0.01:
                 self.entry_price = (total_cost_old - total_cost_new) / self.current_position

        if abs(self.current_position) < 0.01:
            print(f"Position closed.")
            self.entry_price = 0.0
            if self.exit_order_id:
                cancel_order(Ticker.TEAM_A, self.exit_order_id)
                self.exit_order_id = None
        else:
            self._place_profit_taker()

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
                           time_seconds: Optional[float]
        ) -> None:
        if event_type == "END_GAME":
            if self.current_position != 0:
                print("Game ended. Closing open position at market.")
                self._close_position_at_market()
            self.reset_state()
            return

        if event_type == "SCORE":
            score_change = (home_score - self.last_home_score) - (away_score - self.last_away_score)
            if score_change != 0:
                for player in self.home_players_on_court:
                    self.player_ratings[player] = self.player_ratings.get(player, 0) + score_change
                for player in self.away_players_on_court:
                    self.player_ratings[player] = self.player_ratings.get(player, 0) - score_change

            self.last_home_score = home_score
            self.last_away_score = away_score

        if event_type == "SUBSTITUTION" and player_name and substituted_player_name:
            old_lineup_rating = self._calculate_net_lineup_rating()

            target_set = self.home_players_on_court if home_away == "home" else self.away_players_on_court
            if substituted_player_name in target_set:
                target_set.remove(substituted_player_name)
            target_set.add(player_name)

            new_lineup_rating = self._calculate_net_lineup_rating()
            rating_change = new_lineup_rating - old_lineup_rating

            if self.current_position == 0 and abs(rating_change) > self.RATING_CHANGE_THRESHOLD:
                print(f"Significant lineup change detected. Net rating change: {rating_change:.2f}. Placing trade.")
                if rating_change > 0:
                    place_market_order(Side.BUY, Ticker.TEAM_A, self.TRADE_QUANTITY)
                else:
                    place_market_order(Side.SELL, Ticker.TEAM_A, self.TRADE_QUANTITY)

    def _calculate_net_lineup_rating(self) -> float:
        home_rating = sum(self.player_ratings.get(p, 0) for p in self.home_players_on_court)
        away_rating = sum(self.player_ratings.get(p, 0) for p in self.away_players_on_court)
        return home_rating - away_rating

    def _place_profit_taker(self) -> None:
        if self.exit_order_id:
            cancel_order(Ticker.TEAM_A, self.exit_order_id)
            self.exit_order_id = None

        if self.current_position > 0:
            profit_price = self.entry_price + self.PROFIT_TARGET
            self.exit_order_id = place_limit_order(Side.SELL, Ticker.TEAM_A, abs(self.current_position), profit_price)
            print(f"Placed profit-taking SELL order at {profit_price}")
        elif self.current_position < 0:
            profit_price = self.entry_price - self.PROFIT_TARGET
            self.exit_order_id = place_limit_order(Side.BUY, Ticker.TEAM_A, abs(self.current_position), profit_price)
            print(f"Placed profit-taking BUY order at {profit_price}")

    def _close_position_at_market(self):
        if self.exit_order_id:
            cancel_order(Ticker.TEAM_A, self.exit_order_id)
            self.exit_order_id = None

        if self.current_position > 0:
            place_market_order(Side.SELL, Ticker.TEAM_A, abs(self.current_position))
        elif self.current_position < 0:
            place_market_order(Side.BUY, Ticker.TEAM_A, abs(self.current_position))