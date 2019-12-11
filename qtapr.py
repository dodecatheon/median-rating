#!/usr/bin/env python3
import numpy as np
from ballot_tools.csvtoballots import *
from collections import deque
from math import log10
__doc__ = """
QTAR-PR: Quota-Threshold Approval PR.

Run quota-threshold approval (MJ or ER-Bucklin style) to elect
<numseats> winners in a Droop proportional multiwnner election.

Default method is MJ-style, removing votes at the quota-threshold
approval rating until a different rating is found
"""
def droopquota(n,m):
    return(n/(m+1))

def myfmt(x):
    if x > 1:
        fmt = "{:." + "{}".format(int(log10(x))+5) + "g}"
    else:
        fmt = "{:.5g}"
    return fmt.format(x)

def tabulate_score_from_ratings(ballots,weight,maxscore,ncands):
    """tabulate score from ratings"""
    maxscorep1 = maxscore + 1
    S = np.zeros((maxscorep1,ncands))
    for ballot, w in zip(ballots,weight):
        for r in range(1,maxscorep1):
            S[r] += np.where(ballot==r,w,0)
    # Cumulative sum array, S[r:,...].cumsum(axis=0)
    T = np.array([t
                  for t in reversed(np.array([s
                                              for s in reversed(S)]).cumsum(axis=0))])
    return(S,T)
    
def qta(maxscore, quota, ncands, remaining, S, T, use_mj=True, use_two_q=False):
    """Quota Threshold approval single-winner method, using either
    Majority Judgment style tie-breaker for the approval quota
    threshold (default) or ER-Bucklin-ratings style
    """
    ratings = dict()
    twoq = quota * 2

    for c in remaining:
        r1q_unset = True
        r1q = 0
        r2q = 0
        tt_surplus = 0.
        ss = S[...,c]
        tt = T[...,c]
        s = 0
        for r in range(maxscore,-1,-1):
            s += ss[r] * r

            if r1q_unset and (tt[r] > quota):
                r1q_unset = False
                r1q = r
                if not use_two_q:
                    # If not using the two-quota average score tie-breaker,
                    # leading part of the AQT score is the quota threshold rating
                    ratings[c] = (r1q,)
                    break
                
            if tt[r] > twoq:
                r2q = r
                tt_surplus = tt[r2q] - twoq
                break

        if use_two_q:
            # leading part of AQT score is quota threshold rating and average score
            # in the top two quota blocks
            ratings[c] = (r1q, (s - tt_surplus * r) / twoq )
        elif r1q_unset:
            # If not using the two-quota average score tie-breaker,
            # leading part of the AQT score is the quota threshold rating
            ratings[c] = (r1q,)
            
    scores = np.arange(maxscore+1)
    if use_mj:                  # Majority Judgment style approval quota threshold
        for c in remaining:
            ss = S[...,c]
            tt = T[...,c]
            dd = abs(tt - quota)
                
            ratings[c] = (*list(ratings[c]),
                          *[(x,tt[x])
                            for x in np.array(sorted(np.compress(ss>0,scores),
                                                     key=(lambda x:dd[x])))])
    else:                       # ER-Bucklin-ratings style approval quota threshold
        for c in remaining:
            tt = T[...,c]
            ratings[c] = (*list(ratings[c]),
                          tt[r])
            
    ranking = sorted(remaining,key=(lambda c:ratings[c]),reverse=True)
    winner = ranking[0]
    winsum = T[ratings[winner][0],winner]

    if winsum >= quota:
        factor = (1. - quota/winsum)
    else:
        factor = 0
        
    return(winner,winsum,factor,ranking,ratings)

def qtapr(ballots, weights, cnames, numseats,
           verbose=0, use_mj=True, use_two_q=False):
    """Run quota threshold approval rating method (MJ-style or
    Bucklin-style) to elect <numseats> winners in a Droop proportional
    multiwnner election.
    """
    
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
        # Score and Cumulative Score arrays (summing downward from maxscore)
        S, T = tabulate_score_from_ratings(ballots,weights,maxscore,ncands)
        (winner,
         winsum,
         factor,
         ranking,
         ratings) = aqt(maxscore, quota, ncands, cands, S, T,
                        use_mj=use_mj, use_two_q=use_two_q)

        winner_quota_threshold = ratings[winner][0]
        
        # Seat the winner, then eliminate from candidates for next count
        if verbose:
            print("\n-----------\n*** Seat {}: {}\n-----------\n".format(seat+1,cnames[winner]))
            if verbose > 1:
                print("QTAR ranking for this seat:")
                if use_mj:
                    if use_two_q:
                        for c in ranking:
                            r, twoqavg, *rest = ratings[c]
                            print("\t{}:({},{},{})".format(cnames[c],r,myfmt(twoqavg),
                                                           ",".join(["({},{})".format(s,myfmt(t))
                                                                     for s, t in rest])))
                    else:
                        for c in ranking:
                            r, *rest = ratings[c]
                            print("\t{}:({},{})".format(cnames[c],r,
                                                        ",".join(["({},{})".format(s,myfmt(t))
                                                                  for s, t in rest])))
                else:
                    if use_two_q:
                        for c in ranking:
                            r, twoqavg, t = ratings[c]
                            print("\t{}:({},{},{})".format(cnames[c],r,myfmt(twoqavg),myfmt(t)))
                    else:
                        for c in ranking:
                            r, t = ratings[c]
                            print("\t{}:({},{})".format(cnames[c],r,myfmt(t)))
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
            print(("\tWinner's quota approval threshold rating "
                   "before reweighting:  {}%").format(myfmt((winsum/numvotes_orig)*100)))
            print("\tReweighting factor:  ", factor)
            print(("\tPercentage of vote remaining "
                   "after reweighting:  {}%").format(myfmt((numvotes/numvotes_orig)*100)))

    if verbose > 1 and numseats > 1:
        print("- "*30 + "\nReweighting factors for all seat winners:")
        for w, f, qt in zip(winners,factor_array,qthresh_array):
            print("\t{} : ({}, {})".format(cnames[w], myfmt(qt), myfmt(f)))

    if verbose > 3 and numseats > 1:
        print("- "*30 + "\nRemaining ballots and weights:")
        print("{},{}".format("weight",','.join(cnames)))
        for w, ballot in zip(weights,ballots):
            print("{},{}".format(myfmt(w),','.join([str(b) for b in ballot])))

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
                        help="Toggle from MJ-style to ER-Bucklin-style tie-breaker [default: False]")
    parser.add_argument("-q", "--use_two_q", action='store_true',
                        default=False,
                        help="Toggle from quotidian rating to two-quota average score [default: False]")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help="Add verbosity. Can be repeated to increase level. [default: 0]")

    args = parser.parse_args()

    ftype={'score':0, 'rcv':1}[args.filetype]

    ballots, weights, cnames = csvtoballots(args.inputfile,ftype=ftype)

    print("- "*30)
    print("QUOTA-THRESHOLD APPROVAL PROPORTIONAL REPRESENTATION (QTA-PR)")
    if args.verbose > 3:
        print("- "*30)
        # Figure out the width of the weight field, use it to create the format
        ff = '\t{{:{}d}}:'.format(int(log10(weights.max())) + 1)

        print("Ballots:\n\t{}, {}".format('weight',','.join(cnames)))
        for ballot, w in zip(ballots,weights):
            print(ff.format(w),ballot)

    method_type = {True:"ER-Bucklin(ratings)",
                   False:"Majority Judgment"}[args.use_bucklin]
    average_type = {True:"Top-two-quota average score",
                    False:"Quota-centered median rating"}[args.use_two_q]
    print("QTA method:", method_type)
    print("Avg method:", average_type)
    use_mj = not args.use_bucklin
    winners = qtapr(ballots, weights, cnames,
                    args.seats,
                    verbose=args.verbose,
                    use_mj=use_mj,
                    use_two_q=args.use_two_q)
    print("- "*30)

    if args.seats == 1:
        winfmt = "1 winner"
    else:
        winfmt = "{} winners".format(args.seats)

    print("\nMethod:",method_type, ", Averaging:", average_type)
    print("QTAPR returns {}:".format(winfmt),", ".join([cnames[q] for q in winners]))

    return

if __name__ == "__main__":
    main()
