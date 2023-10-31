OBJECTNAV_CURATED_EXAMPLES = [
"""Question: find the sink
Program:

STOP0=False
while not STOP0:
    NAV0=NAVIGATE(goal='sink')
    DET0=DETECT(image=IMAGE, object='sink')
    EVAL0=EVAL(expr='yes' if DET0 else 'no')
    if EVAL0:
        LOC0=LOCATE(pos=POS,object='sink')
        EVAL1=EVAL(expr='STOP' if LOC0 + DET0 > 0 else 'NAVIGATE')
        STOP0=STOP(var=EVAL1)
""",

"""Question: find the red chair in the living room and navigate to it?
STOP0=False # stop if 'living room' is find
STOP1=False # stop if 'red chair' is find
while not STOP0:
    NAV0=NAVIGATE(goal='living room')
    LOC0=LOCATE(pos=POS, object='living room')
    EVAL0=EVAL(expr='yes' if LOC0 else 'no')
    STOP0=STOP(var=EVAL0)
    if EVAL0:
        while not STOP1:
            NAV1=NAVIGATE(goal='red chair')
            DET1=DETECT(image=IMAGE, object='red chair')
            LOC1=LOCATE(pos=POS, object='red chair')
            EVAL1=EVAL(expr='STOP' if LOC1 + DET1 > 0 else 'NAVIGATE')
            STOP1=STOP(var=EVAL1)
""",

"""Question: find the closest window and move to it.
STOP0=False
while not STOP0:
    NAV0=NAVIGATE(goal='closest window')
    DET0=DETECT(image=IMAGE, object='closest window')
    LOC0=LOCATE(pos=POS, object='closest window')
    EVAL0=EVAL(expr='STOP' if LOC0 + DET0 > 0 else 'NAVIGATE')
    STOP0=STOP(var=EVAL0)
"""


]

