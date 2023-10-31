# NAVIGATE makes the agent navigate around unitle stop movement is called in the RL model
# LOCATE returns a boolen value if goal is within range (0.5meters)
# EVAL return 'STOP' if goal is within range, else returns 'NAVIGATE'
# STOP calls the stop action and terminates the episode, returns boolean
# while loop: while STOP is not True, keep navigating

POINTNAV_CURATED_EXAMPLES=[
"""Question: go to the kitchen
Program:
while not STOP0:
    NAV0=NAVIGATE(goal='kitchen')
    LOC0=LOCATE(pos=NAV0)
    EVAL0=EVAL(expr="'STOP' if {LOC0} else 'NAVIGATE'")
    STOP0=STOP(var=EVAL0)
""",

"""Question: go to [10,0,-30]
Program:
while not STOP0:
    NAV0=NAVIGATE(goal='[10,0,-30]')
    LOC0=LOCATE(pos=NAV0)
    EVAL0=EVAL(expr="'STOP' if {LOC0} else 'NAVIGATE'")
    STOP0=STOP(var=EVAL0)
""",

"""Question: go 5 meters forward and 2 meters to the left
Program:
while not STOP0:
    NAV0=NAVIGATE(goal='[5,0,0]')
    LOC0=LOCATE(pos=NAV0)
    EVAL0=EVAL(expr="'STOP' if {LOC0} else 'NAVIGATE'")
    STOP0=STOP(var=EVAL0)
while not STOP1:
    NAV1=NAVIGATE(goal='[0,0,2]')
    LOC1=LOCATE(pos=NAV1)
    EVAL1=EVAL(expr="'STOP' if {LOC1} else 'NAVIGATE'")
    STOP1=STOP(var=EVAL1)
""",

"""Question: go to the bedroom but pass on the bathroom 
Program:
while not STOP0:
    NAV0=NAVIGATE(goal='bathroom')
    LOC0=LOCATE(pos=NAV0)
    EVAL0=EVAL(expr="'STOP' if {LOC0} else 'NAVIGATE'")
    STOP0=STOP(var=EVAL0)
while not STOP1:
    NAV1=NAVIGATE(goal='bedroom')
    LOC1=LOCATE(pos=NAV1)
    EVAL1=EVAL(expr="'STOP' if {LOC1} else 'NAVIGATE'")
    STOP1=STOP(var=EVAL1)
""",

"""Question: go to the sink, then navigate 3 meters on the left 
Program:
while not STOP0:
    NAV0=NAVIGATE(goal='sink')
    LOC0=LOCATE(pos=NAV0)
    EVAL0=EVAL(expr="'STOP' if {LOC0} else 'NAVIGATE'")
    STOP0=STOP(var=EVAL0)
while not STOP1:
    NAV1=NAVIGATE(goal='[0,0,-3]')
    LOC1=LOCATE(pos=NAV1)
    EVAL1=EVAL(expr="'STOP' if {LOC1} else 'NAVIGATE'")
    STOP1=STOP(var=EVAL1)
""",
# ALTERNITAVELY SPLIT ACTIONS: SINK = KITCHEN + SINK
"""Question: go to the sink, then navigate 3 meters on the left 
Program:
while not STOP0:
    NAV0=NAVIGATE(goal='kitchen')
    LOC0=LOCATE(pos=NAV0)
    EVAL0=EVAL(expr="'STOP' if {LOC0} else 'NAVIGATE'")
    STOP0=STOP(var=EVAL0)
while not STOP1:
    NAV1=NAVIGATE(goal='sink')
    LOC1=LOCATE(pos=NAV1)
    EVAL1=EVAL(expr="'STOP' if {LOC1} else 'NAVIGATE'")
    STOP1=STOP(var=EVAL1)
while not STOP2:
    NAV2=NAVIGATE(goal='[0,0,3]')
    LOC2=LOCATE(pos=NAV2)
    EVAL2=EVAL(expr="'STOP' if {LOC2} else 'NAVIGATE'")
    STOP2=STOP(var=EVAL2)
""",

]