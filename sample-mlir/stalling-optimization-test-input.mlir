module {

fsm.machine @cache(){
    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data

    fsm.state @I transitions {
        fsm.transition @store() attributes {nextState=@I_store} {
            %src = fsm.ref @cache
            %dst = fsm.ref @directory
            %msg = fsm.message @Request "GetM" %src, %dst : !fsm.id, !fsm.id -> !fsm.msg
        }

        fsm.transition @load() attributes {nextState=@I_load} {
            %src = fsm.ref @cache
            %dst = fsm.ref @directory
            %msg = fsm.message @Request "GetM" %src, %dst : !fsm.id, !fsm.id -> !fsm.msg
        }
    }

    fsm.state @I_store {prevTransition=@I::@store} transitions {

        fsm.transition @GetM_Ack_D(%msg : !fsm.msg) attributes {nextState = @M}{
            %msg_cl = fsm.access {memberId = "cl"} %msg : !fsm.msg -> !fsm.data
            fsm.update %cl, %msg_cl : !fsm.data
        }

    }

    fsm.state @I_load {prevTransition=@I::@load} transitions {
        fsm.transition @GetM_Ack_D(%msg : !fsm.msg) attributes {nextState = @M}{
            %msg_cl = fsm.access {memberId = "cl"} %msg : !fsm.msg -> !fsm.data
            fsm.update %cl, %msg_cl : !fsm.data
        }
    }

    fsm.state @M transitions {
        fsm.transition @load(){
            fsm.nop
        }

        fsm.transition @store(){
            fsm.nop
        }

        fsm.transition @Fwd_GetM(%Fwd_GetM : !fsm.msg) attributes {nextState = @I}{
            %src = fsm.ref @cache
            %dst = fsm.access {memberId = "src"} %Fwd_GetM : !fsm.msg -> !fsm.id
            %sent_msg = fsm.message @Resp "GetM_Ack_D" %src, %dst, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        }

        fsm.transition @evict() attributes {nextState = @M_evict}{
            %src = fsm.ref @cache
            %dst = fsm.ref @directory
            %msg = fsm.message @Resp "PutM" %src, %dst, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        }
    }

    fsm.state @M_evict {prevTransition = @M::@evict} transitions {
        fsm.transition @Put_Ack(%arg0 : !fsm.msg) attributes {nextState = @I}{
            fsm.nop
        }
    }

}

fsm.machine @directory(){
    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data
    %owner = fsm.variable "owner" : !fsm.id

    fsm.state @I transitions {
        fsm.transition @GetM(%GetM : !fsm.msg) attributes {nextState = @M}{
            %src = fsm.ref @directory
            %dst = fsm.access {memberId = "src"} %GetM : !fsm.msg -> !fsm.id
            %msg = fsm.message @Resp "GetM_Ack_D" %src, %dst, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
            fsm.update %owner, %dst : !fsm.id
        }
    }

    fsm.state @M transitions {
        fsm.transition @GetM(%GetM : !fsm.msg) attributes {nextState = @M}{
            %src = fsm.access {memberId = "src"} %GetM : !fsm.msg -> !fsm.id
            %msg = fsm.message @Request "Fwd_GetM" %src, %owner : !fsm.id, !fsm.id -> !fsm.msg
        }

        fsm.transition @PutM(%PutM : !fsm.msg) {
            %dst = fsm.access {memberId = "src"} %PutM : !fsm.msg -> !fsm.id
            %src = fsm.ref @directory
            %msg = fsm.message @Ack "Put_Ack" %src, %dst : !fsm.id, !fsm.id -> !fsm.msg
            %true = constant true
            fsm.if %true {
                %n_cl = fsm.access {memberId = "cl"} %PutM : !fsm.msg -> !fsm.data
                fsm.update %cl, %n_cl : !fsm.data
                // TODO - figure out how to update state
            }

        }
    }
}

}