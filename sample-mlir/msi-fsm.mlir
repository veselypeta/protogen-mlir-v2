
fsm.machine @cache(){
    %State = fsm.variable {initValue = "I"} : !fsm.state
    %cl = fsm.variable : !fsm.data
    %ackReceived = fsm.variable {initValue = 0} : !fsm.range<0, 3>
    %acksExpected = fsm.variable {initValue = 0} : !fsm.range<0, 3>


}

fsm.machine @directory(){

    %State = fsm.variable {initValue = "I"} : !fsm.state
    %cl = fsm.variable : !fsm.data
    %cache = fsm.variable : !fsm.set<3, !fsm.id>
    %owner = fsm.variable : !fsm.id
}