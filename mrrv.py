#!/usr/bin/env python3
import numpy as np
from ballot_tools.csvtoballots import *
from collections import deque
from math import log10
__doc__ = """
MRRV: Median Rating (droop-quota-based) Reweighted Voting.

Run median ratings method (MJ or ER-Bucklin) to elect <numseats> winners
in a Droop proportional multiwnner election.  Default method is Majority Judgment.
"""
def droopquota(n,m):
    return(n/(m+1))

def myfmt(x):
    if x > 1:
        fmt = "{:." + "{}".format(int(log10(x))+5) + "g}"
    else:
        fmt = "{:.5g}"
    return fmt.format(x)

# Check for qualifying, then if any pass, add them to the ranking
# As qualifying candidates are found, add their (r,s,t) ratings
# to the ratings dict
def find_qualifying_at_rating(r,s,T,MX,remaining,ratings,ranking):
    Q = set([])
    for c in remaining:
        t = T[c]
        mx = MX[c]
        if (mx <= 0) or (t > mx):
            Q.add(c)
            ratings[c] = (r,s,t)

    if len(Q) > 0:
        # Remove all elements of Q from remaining
        remaining.difference_update(Q)
        ranking += sorted(Q,key=(lambda c:T[c]),reverse=True)

def tabulate_score_from_ratings(ballots,weight,maxscore,ncands):
    maxscorep1 = maxscore + 1
    S = np.zeros((maxscorep1,ncands))
    for ballot, w in zip(ballots,weight):
        for r in range(1,maxscorep1):
            S[r] += np.where(ballot==r,w,0)
    return(S)
    
def median_rating(maxscore, quota, ncands, remaining, S,
                  use_mj=True):
    """Returns winner based on median rating method"""
    r = maxscore
    T = np.zeros((ncands))
    MX = np.ones((ncands)) * quota
    ratings = dict()
    ranking = []
    # Ratings significance:
    #
    # ratings[c] = (r,s,t)
    #
    # In each triple (r,s,t) for candidate c:
    #
    # When r is greater than 0,
    # r is the rating level at which
    # "t", the total number of ballots rating candidate c at level s and above
    # exceeds the adjusted quota
    #
    # For ER-Bucklin, r == s and the total t exceeds the quota.
    # For Majority Judgment, there are two tests.  First, whether
    # t exceeds the quota with r == s, and second, whether t exceeds the quota
    # with r reduced by one, s left at previous r, and the quota adjusted downward
    # by S[r,c]/2.  If that qualifying test is passed, it means that
    # the non-quota-passing total at level s is closer to the quota threshold than
    # the quota-passing total at level r.  Then (r,s,t) is the Majority Grade.

    # If r equals zero, then candidate c was not able to pass any qualifying threshold
    # and was sorted at the end of the ranking by total approval.
    while (len(remaining) > 0):
        T += S[r]            # Note that the S[0] all zeros, by construction.
        s = r                # "s" is used to save the ratings level used in T
        find_qualifying_at_rating(r,s,T,MX,remaining,ratings,ranking)

        r -= 1
        if use_mj and (len(remaining) > 0):
            # for Majority Judgment, the adjusted quota is the level at which
            # the unadjusted quota would be exceeded at the new decremented r,
            # but the previous approval count would be closer to the quota than that
            # of the next level down.
            find_qualifying_at_rating(r,s,T,MX - S[r]/2,remaining,ratings,ranking)

        if r == 0:              # Ensure termination
            MX = np.zeros((ncands))
    
    winner = ranking[0]
    r, s, t = ratings[winner]
    winsum = S[r:,winner].sum()

    if winsum >= quota:
        factor = (1. - quota/winsum)
    else:
        factor = 0
        
    return(winner,winsum,factor,ranking,ratings)

def mrrv(ballots, weights, cnames, numseats,
         verbose=0, use_mj=True):
    """Run median ratings method (MJ or Bucklin) to elect <numseats> winners
    in a Droop proportional multiwnner election"""
    
    numballots, numcands = np.shape(ballots)
    ncands = numcands

    numvotes = weights.sum()
    numvotes_orig = float(numvotes)  # Force a copy

    quota = droopquota(numvotes,numseats)

    maxscore = int(ballots.max())

    cands = np.arange(numcands)

    winners = []

    maxscorep1 = maxscore + 1

    factor_array = []
    qthresh_array = []

    for seat in range(numseats):

        if verbose>0:
            print("- "*30,"\nStarting count for seat", seat+1)
            print("Number of votes:",myfmt(numvotes))

        # ----------------------------------------------------------------------
        # Tabulation:
        # ----------------------------------------------------------------------
        S = tabulate_score_from_ratings(ballots,weights,maxscore,ncands)

        # Determine the seat winner using a median rating method:
        remaining = set(cands)
        
        (winner,
         winsum,
         factor,
         ranking,
         ratings) = median_rating(maxscore, quota, ncands, remaining, S,
                                  use_mj=use_mj)

        winner_quota_threshold = ratings[winner][0]
        
        # Seat the winner, then eliminate from candidates for next count
        if verbose:
            print("\n-----------\n*** Seat {}: {}\n-----------\n".format(seat+1,cnames[winner]))
            if verbose > 1:
                print("MR ranking for this seat:")
                for c in ranking:
                    r,s,t = ratings[c]
                    u = S[r:,c].sum()
                    print("\t{}:({},{},{},{})".format(cnames[c],r,s,myfmt(t),myfmt(u)))
                print("")

        if (seat < numseats):
            winners += [winner]
            cands = np.compress(cands != winner,cands)

        weights = np.multiply(weights,
                              np.where(ballots[...,winner] < winner_quota_threshold,
                                       1,
                                       factor))
        numvotes = weights.sum()
        scorerange = np.arange(maxscorep1)

        factor_array.append(factor)
        qthresh_array.append(winner_quota_threshold)

        # Reweight ballots:
        winscores = ballots[...,winner]
        if verbose:
            print("Winner's votes per rating: ",
                  (", ".join(["{}:{}".format(j,myfmt(f))
                              for j, f in zip(scorerange[-1:0:-1],
                                              S[-1:0:-1,winner])])))
            print("After reweighting ballots:")
            print("\tQuota:  {}%".format(myfmt(quota/numvotes_orig*100)))
            print(("\tWinner's approval threshold score "
                   "before reweighting:  {}%").format(myfmt((winsum/numvotes_orig)*100)))
            print("\tReweighting factor:  ", factor)
            print(("\tPercentage of vote remaining "
                   "after reweighting:  {}%").format(myfmt((numvotes/numvotes_orig)*100)))

    if verbose > 1 and numseats > 1:
        print("- "*30 + "\nReweighting factors for all seat winners:")
        for w, f, qt in zip(winners,factor_array,qthresh_array):
            print("\t{} : ({}, {})".format(cnames[w], myfmt(qt), myfmt(f)))

    return(winners)

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-i", "--inputfile",
                        type=str,
                        required=True,
                        help="REQUIRED: CSV Input file [default: none]")
    parser.add_argument("-m", "--seats", type=int,
                        default=1,
                        help="Number of seats [default: 1]")
    parser.add_argument("-t", "--filetype", type=str,
                        choices=["score", "rcv"],
                        default="score",
                        help="CSV file type, either 'score' or 'rcv' [default: 'score']")
    parser.add_argument("-u", "--use_bucklin", action='store_true',
                        default=False,
                        help="Toggle from Majority Judgment to ER-Bucklin [default: False]")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help="Add verbosity. Can be repeated to increase level. [default: 0]")

    args = parser.parse_args()

    ftype={'score':0, 'rcv':1}[args.filetype]

    ballots, weights, cnames = csvtoballots(args.inputfile,ftype=ftype)

    print("- "*30)
    print("MEDIAN RATINGS QUOTA-BASED REWEIGHTED VOTING (MRRV)")
    if args.verbose > 3:
        print("- "*30)
        # Figure out the width of the weight field, use it to create the format
        ff = '\t{{:{}d}}:'.format(int(log10(weights.max())) + 1)

        print("Ballots:\n\t{}, {}".format('weight',','.join(cnames)))
        for ballot, w in zip(ballots,weights):
            print(ff.format(w),ballot)

    method_type = {True:"ER-Bucklin(ratings)",False:"Majority Judgment"}[args.use_bucklin]
    print("MR method:", method_type)
    use_mj = not args.use_bucklin
    winners = mrrv(ballots, weights, cnames,
                   args.seats,
                   verbose=args.verbose,
                   use_mj=use_mj)
    print("- "*30)

    if args.seats == 1:
        winfmt = "1 winner"
    else:
        winfmt = "{} winners".format(args.seats)

    print("\nMethod:",method_type)
    print("MRRV returns {}:".format(winfmt),", ".join([cnames[q] for q in winners]))

    return

if __name__ == "__main__":
    main()
