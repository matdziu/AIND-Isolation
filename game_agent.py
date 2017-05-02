"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self.minimax_recursive(game, game.active_player, 0, depth, True)

    def minimax_recursive(self, game, original_player, current_depth, max_depth, maximizing_player):
        # Check time constraint
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Increase current depth
        current_depth += 1

        # If current recursive node is a leaf node or depth limit is exceeded
        # then return game score based on supplied score function
        if len(game.get_legal_moves(game.active_player)) == 0 or current_depth > max_depth:
            return self.score(game, original_player)

        if maximizing_player:
            # If current player is a maximizing one
            # then recursively compute values of children nodes, take maximum and
            # propagate it up the function calls stack
            moves_with_scores = dict()
            for move in game.get_legal_moves(game.active_player):
                alpha = self.minimax_recursive(game.forecast_move(move), original_player,
                                               current_depth, max_depth, False)
                moves_with_scores[move] = alpha

            # Recognize if it's first function call and return key (move) instead of value
            if current_depth == 1:
                return max(moves_with_scores, key=moves_with_scores.get)
            else:
                return max(moves_with_scores.values())

        else:
            # If current player is a minimizing one
            # then recursively compute values of children nodes, take minimum and
            # propagate it up the function calls stack
            moves_with_scores = dict()
            for move in game.get_legal_moves(game.active_player):
                beta = self.minimax_recursive(game.forecast_move(move), original_player,
                                              current_depth, max_depth, True)
                moves_with_scores[move] = beta

            # Recognize if it's first function call and return key (move) instead of value
            if current_depth == 1:
                return min(moves_with_scores, key=moves_with_scores.get)
            else:
                return min(moves_with_scores.values())

                # Example:
                # A - maximizing, C - minimizing, E - maximizing
                # Maximizing and minimizing players are taking turns
                #
                #                         A
                #                      /     \
                #                    B(10)    C
                #                           /   \
                #                         D(5)   E
                #                              /   \
                #                           F(20)  G(0)
                #
                # minimax(A) -> moves_with_scores = { B: 10, C: minimax(C) }
                # minimax(C) -> moves_with_scores = { D: 5, E: minimax(E) }
                # minimax(E) -> moves_with_scores = { F: 20, G: 0 }
                #
                # Agent propagates upwards maximum or minimum value of each moves_with_scores:
                # minimax(E) -> 20
                # minimax(C) -> moves_with_scores = { D: 5, E: 20 }
                # minimax(A) -> moves_with_scores = { B: 10, D: 5 }
                #
                # Agent recognizes minimax(A) as first function call and returns B which is best move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self.alphabeta_recursive(game, game.active_player, 0, depth, True, alpha, beta)

    def alphabeta_recursive(self, game, original_player, current_depth, max_depth, maximizing_player, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        current_depth += 1

        if current_depth <= max_depth:
            if len(game.get_legal_moves(game.active_player)) == 0:
                return self.score(game, original_player)
        else:
            return self.score(game, original_player)

        if maximizing_player:
            moves_with_scores = dict()
            for move in game.get_legal_moves(game.active_player):
                alpha = self.alphabeta_recursive(game.forecast_move(move), original_player, current_depth,
                                                 max_depth, False, alpha, beta)
                moves_with_scores[move] = alpha

            if current_depth == 1:
                return max(moves_with_scores, key=moves_with_scores.get)
            else:
                return max(moves_with_scores.values())

        else:
            moves_with_scores = dict()
            for move in game.get_legal_moves(game.active_player):
                beta = self.alphabeta_recursive(game.forecast_move(move), original_player, current_depth,
                                                max_depth, True, alpha, beta)
                moves_with_scores[move] = beta

            if current_depth == 1:
                return min(moves_with_scores, key=moves_with_scores.get)
            else:
                return min(moves_with_scores.values())
