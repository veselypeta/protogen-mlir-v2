
fsm machine @cache(){
    %State = fsm.variable {initValue = "I"} : !fsm.state
    %cl = fsm.variable : !fsm.data
    %ackReceived = fsm.variable {initValue = 0} : !fsm.range<0, 3>

}