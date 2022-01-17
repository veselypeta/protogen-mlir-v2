module {

fsm.machine @cache(){
    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data

    fsm.state @I transitions {
        fsm.transition @store() attributes {nextState=@I_store} {
            %msg = fsm.message @Request "GetM" -> !fsm.msg
            // send on correct network
        }

        fsm.transition @load() attributes {nextState=@I_load} {
            %msg = fsm.message @Request "GetM" -> !fsm.msg
        }
    }

    fsm.state @I_store transitions {

        fsm.transition @GetM_Ack_D(%msg : !fsm.msg){
            %msg_cl = fsm.access {memberId = "cl"} %msg : !fsm.msg -> !fsm.data
            fsm.update %cl, %msg_cl : !fsm.data
        }

    }


}

fsm.machine @directory(){
    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data
    %owner = fsm.variable "owner" : !fsm.id
}

}