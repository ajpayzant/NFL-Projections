# How the projections work

Every QB, RB, WR and TE expected to play this week gets a projection for his passing, rushing and
receiving line, his chance of scoring, and his fantasy points; every game gets a fair moneyline,
spread and total. All of it is checked walk-forward: each season projected with models fitted only
on the seasons before it.

## 1. Who plays

The first question is who will be on the field. For every rostered QB, RB, WR and TE the model
estimates the chance he records a snap, and for each team the likeliest starting quarterback, from
his recent snaps, his place on the depth chart, how often he has played lately, and the injury report.

- Before the week's injury report is out (it is published Wednesday to Friday) the model uses last
  week's report and whether he was inactive; once it is out, this week's.
- It names who plays correctly about 92% of the time, and the starting quarterback 90-95%.
- The **Depth charts** tab overrides any of it: rule a player out or back in, or change the starter.

## 2. The football chain

The projections come from a model of how a game produces statistics rather than from the box
score directly: how many plays a team runs and how many are passes, how much of the field time and
how many routes each player has, his share of the targets and carries, and what he does with each
one. So a player's line always fits his team's: the targets of a team's receivers add up to what its
quarterback is projected to throw.

- Team volume reads the opponent, pace, rest, weather and the betting market's total and spread.
- The players' shares are measured against the teammates who will actually play this week.
- A second model, fitted directly on each statistic, is blended in where it measured better
  (backs' carries, receivers' targets, touchdowns); the engine keeps the quarterback's volume.

## 3. Probabilities

A projection is the average outcome; a prop asks how often he goes over a line. The spread of
outcomes is read from history: for counts (receptions, targets, carries, attempts) a fitted count
distribution, and for yards, touchdowns and fantasy points the actual spread of results of players
with similar projections, which prices the lines books post to within about half a point. The chance
of a touchdown is calibrated so it keeps rising for the busiest backs. Every probability is for a game
he plays (books void a prop otherwise).

## 4. Game lines

The game lines use **no betting-market input**, so they can disagree with the books: a regression on
both teams' points, efficiency and success rates per play, pace, rest, weather and quarterback and coach
continuity. Margins and totals are priced on the distribution of real NFL results, so the key numbers
are kept: a game projected near 3 lands on exactly 3 about 9% of the time.

## 5. How accurate it is

| | Model | Market |
|---|---|---|
| Game margin, average miss (2019-26) | 10.2 pts | 9.8 pts |
| Winners picked | 64.7% | 66.2% |
| Total, average miss | 10.7 pts | 10.4 pts |

The market is better on game lines, as it should be (it knows late news); a model that matched it would
have no edges to find. The player chain beats a player's own recent averages on 22 of 23 statistics,
and the blend improves on the chain on 16 of 19.

## 6. Using it

- **Slate:** the week's game lines and every player by position, with prop probabilities or fair odds.
- **Game:** one game's board and both teams by position.
- **Depth charts:** who plays and who starts at quarterback, with your edits.
- **Player:** one player's projection, prop ladder with fair odds, fantasy range and game log.
- **Edges:** sportsbook prices against the model by position; odds are fetched only when you click
  Refresh odds. "Check a line from a book" prices any line you type in.
- **Fantasy:** PPR, half-PPR and standard projections with a floor and ceiling for every player.
- **Results:** the model graded against the opening and closing lines.

A big edge is more often news the model has not seen (an injury, a depth-chart change) than a
mispriced line: check the injury report before playing it.
