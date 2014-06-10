from stats import compareLists

def globalResult(real_ref, lead, max_error_margin = 10, points  = False):
    beats = []
    for x in lead:
        beats = beats + x[2]

    beats = sorted(list(set(beats)))

    return compareLists(real_ref, beats,max_error_margin, points)
