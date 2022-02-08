%fwd = fsm.network @fwd "ordered"
%resp = fsm.network @resp "unordered"
%req = fsm.network @req "unordered"

fsm.m_decl @Request decls  {
    fsm.nop
}

fsm.m_decl @Ack decls  {
    fsm.nop
}

fsm.m_decl @Resp decls  {
    %cl = fsm.m_var @cl : !fsm.data
}

fsm.m_decl @RespAck decls {
    %cl = fsm.m_var @cl : !fsm.data
    %acksExpected = fsm.m_var @acksExpected : !fsm.range<0,3>
}



fsm.machine @cache(){
    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data
    %acksReceived = fsm.variable "acksReceived" {initValue = 0} : !fsm.range<0, 3>
    %acksExpected = fsm.variable "acksExpected" {initValue = 0} : !fsm.range<0, 3>

    fsm.state @I transitions {

        fsm.transition @load() attributes {nextState=@I_load} {
            %src = fsm.ref @cache
            %dst = fsm.ref @directory
            %msg = fsm.message @Request "GetS" %src, %dst : !fsm.id, !fsm.id -> !fsm.msg
        }

        fsm.transition @store() attributes {nextState=@I_store} {
            %src = fsm.ref @cache
            %dst = fsm.ref @directory
            %msg = fsm.message @Request "GetM" %src, %dst : !fsm.id, !fsm.id -> !fsm.msg
            %n_cnt = fsm.constant {value = "0"} : i64
            fsm.update %acksReceived, %n_cnt : !fsm.range<0, 3>, i64
        }
    } // end I

    fsm.state @I_load {prevTransition=@I::@load} transitions {

        fsm.transition @GetS_Ack(%msg : !fsm.msg) attributes {nextState=@S}{
            %n_cl = fsm.access {memberId = "cl"} %msg : !fsm.msg -> !fsm.data
            fsm.update %cl, %n_cl : !fsm.data, !fsm.data
        }

    } // end I_load


    fsm.state @I_store {prevTransition=@I::@store} transitions {

        fsm.transition @GetM_Ack_D(%msg : !fsm.msg) attributes {nextState=@M}{
            %n_cl = fsm.access {memberId = "cl"} %msg : !fsm.msg -> !fsm.data
            fsm.update %cl, %n_cl : !fsm.data, !fsm.data
        }

        fsm.transition @GetM_Ack_AD(%msg : !fsm.msg) attributes {nextState=@I_store_GetM_Ack_AD}{
            %e_ack = fsm.access {memberId = "acksExpected"} %msg : !fsm.msg -> !fsm.range<0,3>
            fsm.update %acksExpected, %e_ack : !fsm.range<0,3>, !fsm.range<0,3>
        }

        fsm.transition @Inv_Ack() attributes {nextState=@I_store}{
            %c_1 = fsm.constant {value = "1"} : i64
            %inc = fsm.add %acksReceived, %c_1 : !fsm.range<0, 3>, i64
            fsm.update %acksReceived, %inc : !fsm.range<0, 3>, i64
        }
    } // end I_store

    fsm.state @I_store_GetM_Ack_AD {prevTransition=@I_store::@GetM_Ack_AD} transitions {
        fsm.transition @Inv_Ack(%msg : !fsm.msg) {
            %c_1 = fsm.constant {value = "1"} : i64
            %inc = fsm.add %acksReceived, %c_1 : !fsm.range<0, 3>, i64
            fsm.update %acksReceived, %inc : !fsm.range<0, 3>, i64

            %is_eq = fsm.comp "=" %acksExpected, %acksReceived : !fsm.range<0,3>, !fsm.range<0,3>
            fsm.if %is_eq {
                %m_state = fsm.constant {value="M"} : !fsm.state
                fsm.update %State, %m_state : !fsm.state, !fsm.state
            }
        }
    } // end I_store_GetM_Ack_AD

    fsm.state @S transitions {
        fsm.transition @load() attributes {nextState = @S}{
            fsm.nop
        }
        fsm.transition @store() attributes {nextState = @S_store}{
            %src = fsm.ref @cache
            %dst = fsm.ref @directory
            %msg = fsm.message @Request "Upgrade" %src, %dst : !fsm.id, !fsm.id -> !fsm.msg
            fsm.send %req %msg
            %c_0 = fsm.constant {value="0"} : i64
            fsm.update %acksReceived, %c_0 : !fsm.range<0, 3>, i64
        }
        fsm.transition @evict() attributes {nextState=@S_evict} {
            %src = fsm.ref @cache
            %dst = fsm.ref @directory
            %msg = fsm.message @Request "PutS" %src, %dst : !fsm.id, !fsm.id -> !fsm.msg
        }
        fsm.transition @Inv(%msg : !fsm.msg) attributes {nextState=@I}{
            %src = fsm.ref @cache
            %dst = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %n_msg = fsm.message @Resp "Inv_Ack" %src, %dst, %cl: !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
            fsm.send %resp %n_msg
        }
    } // end S

    fsm.state @S_store {prevTransition=@S::@store} transitions {

        fsm.transition @GetM_Ack_AD(%msg : !fsm.msg) attributes {nextState=@S_store_GetM_Ack_AD} {
            %r_acks = fsm.access {memberId = "acksExpected"} %msg : !fsm.msg -> !fsm.range<0,3>
            fsm.update %acksExpected, %r_acks : !fsm.range<0,3>,!fsm.range<0,3>
            %is_eq = fsm.comp "=" %acksExpected, %acksReceived : !fsm.range<0,3>, !fsm.range<0,3>
            fsm.if %is_eq {
                %n_state = fsm.constant {value = "M"} : !fsm.state
                fsm.update %State, %n_state : !fsm.state, !fsm.state
            }
        }

        fsm.transition @Inv_Ack(%msg : !fsm.msg) attributes {nextState=@S_store} {
            %c_1 = fsm.constant {value="1"} : i64
            %n_v = fsm.add %c_1, %acksReceived : i64, !fsm.range<0, 3>
            fsm.update %acksReceived, %n_v : !fsm.range<0,3>, i64
        }

    } // end S_store

    fsm.state @S_store_GetM_Ack_AD {prevTransition=@S_store::@GetM_Ack_AD} transitions {
        fsm.transition @Inv_Ack(%msg : !fsm.msg) {
            %c_1 = fsm.constant {value="1"} : i64
            %n_v = fsm.add %c_1, %acksReceived : i64, !fsm.range<0, 3>
            fsm.update %acksReceived, %n_v : !fsm.range<0,3>, i64

            %is_eq = fsm.comp "=" %acksExpected, %acksReceived : !fsm.range<0,3>, !fsm.range<0,3>
            fsm.if %is_eq {
                %n_state = fsm.constant {value = "M"} : !fsm.state
                fsm.update %State, %n_state : !fsm.state, !fsm.state
            }
        }

    } // end S_store_GetM_Ack_AD

    fsm.state @S_evict {prevTransition=@S::@evict} transitions {
        fsm.transition @Put_Ack(%msg : !fsm.msg) attributes {nextState=@I} {
            fsm.nop
        }
    } // end S_evict

    fsm.state @M transitions {
        fsm.transition @load(){
            fsm.nop
        }

        fsm.transition @store(){
            fsm.nop
        }

        fsm.transition @Fwd_GetM(%msg : !fsm.msg) attributes {nextState=@I} {
            %src = fsm.ref @cache
            %dst = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %n_msg = fsm.message @Resp "GetM_Ack_D" %src, %dst, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
            fsm.send %resp %n_msg
        }

        fsm.transition @Fwd_GetS(%msg : !fsm.msg) attributes {nextState=@S} {
            %src1 = fsm.ref @cache
            %dst1 = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %n_msg1 = fsm.message @Resp "GetS_Ack" %src1, %dst1, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
            fsm.send %resp %n_msg1

            %src2 = fsm.ref @cache
            %dst2 = fsm.ref @directory
            %n_msg2 = fsm.message @Resp "WB" %src2, %dst2, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
            fsm.send %resp %n_msg2
        }

        fsm.transition @evict() attributes {nextState=@M_evict} {
            %src = fsm.ref @cache
            %dst = fsm.ref @directory
            %p_msg = fsm.message @Resp "PutM" %src, %dst, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
            fsm.send %req %p_msg
        }
    } // end M

    fsm.state @M_evict {prevTransition=@M::@evict} transitions {
        fsm.transition @PutAck(%msg : !fsm.msg) attributes {nextState = @I}{
            fsm.nop
        }
    }
}

fsm.machine @directory(){
    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data
    %cache = fsm.variable "cache" : !fsm.set<!fsm.id, 3>
    %owner = fsm.variable "owner" : !fsm.id

    fsm.state @I transitions {
        fsm.transition @GetS(%msg : !fsm.msg) attributes {nextState=@S}{
            %src = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            fsm.set_add %cache, %src : !fsm.set<!fsm.id, 3>, !fsm.id

            %msg_src = fsm.ref @directory
            %msg_dst = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %n_msg = fsm.message @Resp "GetS_Ack" %msg_src, %msg_dst, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
            fsm.send %resp %n_msg
        }
        fsm.transition @GetM(%msg : !fsm.msg) attributes {nextState=@M}{
            %m_src = fsm.ref @directory
            %m_dst = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %m_cnt = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
            %n_msg = fsm.message @RespAck "GetM_Ack_AD" %m_src, %m_dst, %cl, %m_cnt : !fsm.id, !fsm.id, !fsm.data, i64 -> !fsm.msg

            %n_own = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            fsm.update %owner, %n_own : !fsm.id, !fsm.id
        }
    } // end I

    fsm.state @S transitions {

        fsm.transition @GetS(%msg : !fsm.msg) attributes {nextState=@S} {
            %src = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            fsm.set_add %cache, %src : !fsm.set<!fsm.id, 3>, !fsm.id

            %m_src = fsm.ref @directory
            %m_dst = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %n_msg = fsm.message @Resp "GetS_Ack" %m_src, %m_dst, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
            fsm.send %resp %n_msg
        }

        fsm.transition @Upgrade(%msg : !fsm.msg) {
            %src = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %contains = fsm.set_contains %cache, %src : !fsm.set<!fsm.id, 3>, !fsm.id
            fsm.if %contains {
                %u_src = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
                fsm.set_delete %cache, %u_src : !fsm.set<!fsm.id, 3>, !fsm.id
                %c_cnt = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
                %self = fsm.ref @directory
                %s_msg = fsm.message @RespAck "GetM_Ack_AD" %self, %u_src, %cl, %c_cnt : !fsm.id, !fsm.id, !fsm.data, i64 -> !fsm.msg
                fsm.send %resp %s_msg
                %m_state = fsm.constant {value = "M"} : !fsm.state
                fsm.update %State, %m_state : !fsm.state, !fsm.state
            } else {
                %u_src = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
                %c_cnt = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
                %self = fsm.ref @directory
                %s_msg = fsm.message @RespAck "GetM_Ack_AD" %self, %u_src, %cl, %c_cnt : !fsm.id, !fsm.id, !fsm.data, i64 -> !fsm.msg
                fsm.send %resp %s_msg
                %m_state = fsm.constant {value = "M"} : !fsm.state
                fsm.update %State, %m_state : !fsm.state, !fsm.state
            }
            %ack_m = fsm.message @Ack "Inv" %src, %src : !fsm.id, !fsm.id -> !fsm.msg
            fsm.multicast %fwd %ack_m %cache : !fsm.set<!fsm.id, 3>
            fsm.update %owner, %src : !fsm.id, !fsm.id
            fsm.set_clear %cache : !fsm.set<!fsm.id, 3>
        }

        fsm.transition @PutS(%msg : !fsm.msg){
            %PutSSrc = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %src = fsm.ref @directory
            %PutAck = fsm.message @Resp "Put_Ack" %src, %PutSSrc, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
            fsm.send %fwd %PutAck
            fsm.set_delete %cache, %PutSSrc : !fsm.set<!fsm.id, 3>, !fsm.id

            %c_count = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
            %zero = fsm.constant {value = "0"} : i64
            %eq_zero = fsm.comp "=" %c_count, %zero : i64, i64
            fsm.if %eq_zero {
                %I_state = fsm.constant {value = "I"} : !fsm.state
                fsm.update %State, %I_state : !fsm.state, !fsm.state
            }
        }
    } // end S


    fsm.state @M transitions {
        fsm.transition @GetS(%msg : !fsm.msg) attributes {nextState=@M_GetS}{
            %GetSSrc = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %Fwd_GetS = fsm.message @Request "Fwd_GetS" %GetSSrc, %owner : !fsm.id, !fsm.id -> !fsm.msg
            fsm.set_add %cache, %GetSSrc : !fsm.set<!fsm.id, 3>, !fsm.id
            fsm.set_add %cache, %owner : !fsm.set<!fsm.id, 3>, !fsm.id
        }

        fsm.transition @GetM(%msg : !fsm.msg) {
            %GetMSrc = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %Fwd_GetM = fsm.message @Request "Fwd_GetM" %GetMSrc, %owner : !fsm.id, !fsm.id -> !fsm.msg
            fsm.send %fwd %msg
            fsm.update %owner, %GetMSrc : !fsm.id, !fsm.id
        }

        fsm.transition @PutM(%msg : !fsm.msg) {
            %PutMSrc = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %src = fsm.ref @directory
            %Ack = fsm.message @Ack "Put_Ack" %src, %PutMSrc : !fsm.id, !fsm.id -> !fsm.msg
            fsm.send %fwd %Ack
            fsm.set_delete %cache, %PutMSrc : !fsm.set<!fsm.id, 3>, !fsm.id

            %is_eq = fsm.comp "=" %owner, %PutMSrc : !fsm.id, !fsm.id
            fsm.if %is_eq {
                %PutMcl = fsm.access {memberId = "cl"} %msg : !fsm.msg -> !fsm.data
                fsm.update %cl, %PutMcl : !fsm.data, !fsm.data

                %I_state = fsm.constant {value = "I"} : !fsm.state
                fsm.update %State, %I_state : !fsm.state, !fsm.state
            }
        }

    } // end M

    fsm.state @M_GetS transitions {
        fsm.transition @WB(%msg : !fsm.msg){
            %WBSrc = fsm.access {memberId = "src"} %msg : !fsm.msg -> !fsm.id
            %is_eq = fsm.comp "=" %WBSrc, %owner : !fsm.id, !fsm.id
            fsm.if %is_eq {
                %WBcl = fsm.access {memberId = "cl"} %msg : !fsm.msg -> !fsm.data
                fsm.update %cl, %WBcl : !fsm.data, !fsm.data
                %S_state = fsm.constant {value = "S"} : !fsm.state
                fsm.update %State, %S_state : !fsm.state, !fsm.state
            }
        }
    } // end S_GetS




}