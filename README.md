# Quota Threshold Approval
An extension of Majority Choice Approval to multiwinner PR elections

## Quota Threshold
For each seat, the quota threshold is found as follows:

* Initialize the quota threshold to the maxscore
* For each candidate, add in the weighted total of ballots scoring that
  candidate at the QT rating.
* If at least one candidate has a total greater than the quota, use
  tie-breaker to resolve the seat winner.
* Exhaust one quota's weight from ballots that score the seat winner at or
  above the quota threshold rating.

## Tie-breaking
When more than one candidate satisfies the quota threshold for a seat, several
tie-breakers may be used:
* Bucklin style: the candidate with the greatest weighted total at or above
  the quota threshold rating.  This method reduces to MCA-M (AKA ER-Bucklin)
  in the single winner case.
* Majority Judgment style: when removing votes for each candidate at the quota
  threshold rating, the candidate who switches to a higher rating, or doesn't
  drop to a lower rating, before other candidates, is the seat winner.
