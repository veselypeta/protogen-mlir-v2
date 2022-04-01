module  {
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
  fsm.m_decl @RespAck decls  {
    %cl = fsm.m_var @cl : !fsm.data
    %acksExpected = fsm.m_var @acksExpected : !fsm.range<0, 3>
  }
  fsm.machine @cache() {
    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data
    %acksReceived = fsm.variable "acksReceived" {initValue = 0 : i64} : !fsm.range<0, 3>
    %acksExpected = fsm.variable "acksExpected" {initValue = 0 : i64} : !fsm.range<0, 3>
    fsm.state @I transitions  {
      fsm.transition @load() attributes {nextState = @I_load} {
        %0 = fsm.ref @cache
        %1 = fsm.ref @directory
        %2 = fsm.message @Request "GetS" %0, %1 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %req %2
      }
      fsm.transition @store() attributes {nextState = @I_store} {
        %0 = fsm.ref @cache
        %1 = fsm.ref @directory
        %2 = fsm.message @Request "GetM" %0, %1 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %req %2
        %3 = fsm.constant {value = "0"} : i64
        fsm.update %acksReceived, %3 : !fsm.range<0, 3>, i64
      }
    }
    fsm.state @I_load {prevTransition = @I::@load} transitions  {
      fsm.transition @GetS_Ack(%arg0: !fsm.msg) attributes {nextState = @S} {
        %0 = fsm.access {memberId = "cl"} %arg0 : !fsm.msg -> !fsm.data
        fsm.update %cl, %0 : !fsm.data, !fsm.data
      }
    }
    fsm.state @I_store {prevTransition = @I::@store} transitions  {
      fsm.transition @GetM_Ack_D(%arg0: !fsm.msg) attributes {nextState = @M} {
        %0 = fsm.access {memberId = "cl"} %arg0 : !fsm.msg -> !fsm.data
        fsm.update %cl, %0 : !fsm.data, !fsm.data
      }
      fsm.transition @GetM_Ack_AD(%arg0: !fsm.msg) {
        %0 = fsm.access {memberId = "acksExpected"} %arg0 : !fsm.msg -> !fsm.range<0, 3>
        fsm.update %acksExpected, %0 : !fsm.range<0, 3>, !fsm.range<0, 3>
        %1 = fsm.comp "=" %acksExpected, %acksReceived : !fsm.range<0, 3>, !fsm.range<0, 3>
        fsm.if %1 {
          %2 = fsm.constant {value = "M"} : !fsm.state
          fsm.update %State, %2 : !fsm.state, !fsm.state
        } else {
          %2 = fsm.constant {value = "I_store_GetM_Ack_AD"} : !fsm.state
          fsm.update %State, %2 : !fsm.state, !fsm.state
        }
      }
      fsm.transition @Inv_Ack() attributes {nextState = @I_store} {
        %0 = fsm.constant {value = "1"} : i64
        %1 = fsm.add %acksReceived, %0 : !fsm.range<0, 3>, i64
        fsm.update %acksReceived, %1 : !fsm.range<0, 3>, i64
      }
    }
    fsm.state @I_store_GetM_Ack_AD {prevTransition = @I_store::@GetM_Ack_AD} transitions  {
      fsm.transition @Inv_Ack(%arg0: !fsm.msg) {
        %0 = fsm.constant {value = "1"} : i64
        %1 = fsm.add %acksReceived, %0 : !fsm.range<0, 3>, i64
        fsm.update %acksReceived, %1 : !fsm.range<0, 3>, i64
        %2 = fsm.comp "=" %acksExpected, %acksReceived : !fsm.range<0, 3>, !fsm.range<0, 3>
        fsm.if %2 {
          %3 = fsm.constant {value = "M"} : !fsm.state
          fsm.update %State, %3 : !fsm.state, !fsm.state
        }
      }
    }
    fsm.state @S transitions  {
      fsm.transition @load() attributes {nextState = @S} {
        fsm.nop
      }
      fsm.transition @store() attributes {nextState = @S_store} {
        %0 = fsm.ref @cache
        %1 = fsm.ref @directory
        %2 = fsm.message @Request "Upgrade" %0, %1 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %req %2
        %3 = fsm.constant {value = "0"} : i64
        fsm.update %acksReceived, %3 : !fsm.range<0, 3>, i64
      }
      fsm.transition @evict() attributes {nextState = @S_evict} {
        %0 = fsm.ref @cache
        %1 = fsm.ref @directory
        %2 = fsm.message @Request "PutS" %0, %1 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %req %2
      }
      fsm.transition @Inv(%arg0: !fsm.msg) attributes {nextState = @I} {
        %0 = fsm.ref @cache
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.message @Resp "Inv_Ack" %0, %1, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %2
      }
    }
    fsm.state @S_store {prevTransition = @S::@store} transitions  {
      fsm.transition @GetM_Ack_AD(%arg0: !fsm.msg) {
        %0 = fsm.access {memberId = "acksExpected"} %arg0 : !fsm.msg -> !fsm.range<0, 3>
        fsm.update %acksExpected, %0 : !fsm.range<0, 3>, !fsm.range<0, 3>
        %1 = fsm.comp "=" %acksExpected, %acksReceived : !fsm.range<0, 3>, !fsm.range<0, 3>
        fsm.if %1 {
          %2 = fsm.constant {value = "M"} : !fsm.state
          fsm.update %State, %2 : !fsm.state, !fsm.state
        } else {
          %2 = fsm.constant {value = "S_store_GetM_Ack_AD"} : !fsm.state
          fsm.update %State, %2 : !fsm.state, !fsm.state
        }
      }
      fsm.transition @Inv_Ack(%arg0: !fsm.msg) attributes {nextState = @S_store} {
        %0 = fsm.constant {value = "1"} : i64
        %1 = fsm.add %0, %acksReceived : i64, !fsm.range<0, 3>
        fsm.update %acksReceived, %1 : !fsm.range<0, 3>, i64
      }
      fsm.transition @Inv(%arg0: !fsm.msg) attributes {nextState = @I_store} {
        %0 = fsm.ref @cache
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.message @Resp "Inv_Ack" %0, %1, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %2
      }
    }
    fsm.state @S_store_GetM_Ack_AD {prevTransition = @S_store::@GetM_Ack_AD} transitions  {
      fsm.transition @Inv_Ack(%arg0: !fsm.msg) {
        %0 = fsm.constant {value = "1"} : i64
        %1 = fsm.add %0, %acksReceived : i64, !fsm.range<0, 3>
        fsm.update %acksReceived, %1 : !fsm.range<0, 3>, i64
        %2 = fsm.comp "=" %acksExpected, %acksReceived : !fsm.range<0, 3>, !fsm.range<0, 3>
        fsm.if %2 {
          %3 = fsm.constant {value = "M"} : !fsm.state
          fsm.update %State, %3 : !fsm.state, !fsm.state
        }
      }
      fsm.transition @Inv(%arg0: !fsm.msg) attributes {nextState = @I_store} {
        %0 = fsm.ref @cache
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.message @Resp "Inv_Ack" %0, %1, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %2
      }
    }
    fsm.state @S_evict {prevTransition = @S::@evict} transitions  {
      fsm.transition @Put_Ack(%arg0: !fsm.msg) attributes {nextState = @I} {
        fsm.nop
      }
      fsm.transition @Inv(%arg0: !fsm.msg) attributes {nextState = @I_PutS} {
        %0 = fsm.ref @cache
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.message @Resp "Inv_Ack" %0, %1, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %2
      }
    }
    fsm.state @I_PutS {prevTransition = @Inv} transitions  {
      fsm.transition @Put_Ack(%arg0: !fsm.msg) attributes {nextState = @I} {
        fsm.nop
      }
    }
    fsm.state @M transitions  {
      fsm.transition @load() {
        fsm.nop
      }
      fsm.transition @store() {
        fsm.nop
      }
      fsm.transition @Fwd_GetM(%arg0: !fsm.msg) attributes {nextState = @I} {
        %0 = fsm.ref @cache
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.message @Resp "GetM_Ack_D" %0, %1, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %2
      }
      fsm.transition @Fwd_GetS(%arg0: !fsm.msg) attributes {nextState = @S} {
        %0 = fsm.ref @cache
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.message @Resp "GetS_Ack" %0, %1, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %2
        %3 = fsm.ref @cache
        %4 = fsm.ref @directory
        %5 = fsm.message @Resp "WB" %3, %4, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %5
      }
      fsm.transition @evict() attributes {nextState = @M_evict} {
        %0 = fsm.ref @cache
        %1 = fsm.ref @directory
        %2 = fsm.message @Resp "PutM" %0, %1, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %req %2
      }
    }
    fsm.state @M_evict {prevTransition = @M::@evict} transitions  {
      fsm.transition @Put_Ack(%arg0: !fsm.msg) attributes {nextState = @I} {
        fsm.nop
      }
      fsm.transition @Fwd_GetM(%arg0: !fsm.msg) attributes {nextState = @I_PutM} {
        %0 = fsm.ref @cache
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.message @Resp "GetM_Ack_D" %0, %1, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %2
      }
      fsm.transition @Fwd_GetS(%arg0: !fsm.msg) attributes {nextState = @S_evict} {
        %0 = fsm.ref @cache
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.message @Resp "GetS_Ack" %0, %1, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %2
        %3 = fsm.ref @cache
        %4 = fsm.ref @directory
        %5 = fsm.message @Resp "WB" %3, %4, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %5
      }
    }
    fsm.state @I_PutM {prevTransition = @Fwd_GetM} transitions  {
      fsm.transition @Put_Ack(%arg0: !fsm.msg) attributes {nextState = @I} {
        fsm.nop
      }
    }
  }
  fsm.machine @directory() {
    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data
    %cache = fsm.variable "cache" : !fsm.set<!fsm.id, 3>
    %owner = fsm.variable "owner" : !fsm.id
    fsm.state @I transitions  {
      fsm.transition @GetS(%arg0: !fsm.msg) attributes {nextState = @S} {
        %0 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.set_add %cache, %0 : !fsm.set<!fsm.id, 3>, !fsm.id
        %1 = fsm.ref @directory
        %2 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %3 = fsm.message @Resp "GetS_Ack" %1, %2, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %3
      }
      fsm.transition @GetM(%arg0: !fsm.msg) attributes {nextState = @M} {
        %0 = fsm.ref @directory
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
        %3 = fsm.message @RespAck "GetM_Ack_AD" %0, %1, %cl, %2 : !fsm.id, !fsm.id, !fsm.data, i64 -> !fsm.msg
        fsm.send %resp %3
        %4 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.update %owner, %4 : !fsm.id, !fsm.id
      }
    }
    fsm.state @S transitions  {
      fsm.transition @GetS(%arg0: !fsm.msg) attributes {nextState = @S} {
        %0 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.set_add %cache, %0 : !fsm.set<!fsm.id, 3>, !fsm.id
        %1 = fsm.ref @directory
        %2 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %3 = fsm.message @Resp "GetS_Ack" %1, %2, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %3
      }
      fsm.transition @Upgrade(%arg0: !fsm.msg) attributes {nextState = @M} {
        %0 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %1 = fsm.set_contains %cache, %0 : !fsm.set<!fsm.id, 3>, !fsm.id
        fsm.if %1 {
          %3 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
          fsm.set_delete %cache, %3 : !fsm.set<!fsm.id, 3>, !fsm.id
          %4 = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
          %5 = fsm.ref @directory
          %6 = fsm.message @RespAck "GetM_Ack_AD" %5, %3, %cl, %4 : !fsm.id, !fsm.id, !fsm.data, i64 -> !fsm.msg
          fsm.send %resp %6
          %7 = fsm.constant {value = "M"} : !fsm.state
          fsm.update %State, %7 : !fsm.state, !fsm.state
        } else {
          %3 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
          %4 = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
          %5 = fsm.ref @directory
          %6 = fsm.message @RespAck "GetM_Ack_AD" %5, %3, %cl, %4 : !fsm.id, !fsm.id, !fsm.data, i64 -> !fsm.msg
          fsm.send %resp %6
          %7 = fsm.constant {value = "M"} : !fsm.state
          fsm.update %State, %7 : !fsm.state, !fsm.state
        }
        %2 = fsm.message @Ack "Inv" %0, %0 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.multicast %fwd %2 %cache : !fsm.set<!fsm.id, 3>
        fsm.update %owner, %0 : !fsm.id, !fsm.id
        fsm.set_clear %cache : !fsm.set<!fsm.id, 3>
      }
      fsm.transition @PutS(%arg0: !fsm.msg) {
        %0 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %1 = fsm.ref @directory
        %2 = fsm.message @Resp "Put_Ack" %1, %0, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %fwd %2
        fsm.set_delete %cache, %0 : !fsm.set<!fsm.id, 3>, !fsm.id
        %3 = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
        %4 = fsm.constant {value = "0"} : i64
        %5 = fsm.comp "=" %3, %4 : i64, i64
        fsm.if %5 {
          %6 = fsm.constant {value = "I"} : !fsm.state
          fsm.update %State, %6 : !fsm.state, !fsm.state
        }
      }
      fsm.transition @PutM(%arg0: !fsm.msg) {
        fsm.call @PutS(%arg0)
      }
    }
    fsm.state @M transitions  {
      fsm.transition @GetS(%arg0: !fsm.msg) attributes {nextState = @M_GetS} {
        %0 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %1 = fsm.message @Request "Fwd_GetS" %0, %owner : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %fwd %1
        fsm.set_add %cache, %0 : !fsm.set<!fsm.id, 3>, !fsm.id
        fsm.set_add %cache, %owner : !fsm.set<!fsm.id, 3>, !fsm.id
      }
      fsm.transition @GetM(%arg0: !fsm.msg) attributes {nextState = @M} {
        %0 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %1 = fsm.message @Request "Fwd_GetM" %0, %owner : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %fwd %1
        fsm.update %owner, %0 : !fsm.id, !fsm.id
      }
      fsm.transition @Upgrade(%arg0: !fsm.msg) attributes {nextState = @M} {
        fsm.call @GetM(%arg0)
      }
      fsm.transition @PutM(%arg0: !fsm.msg) {
        %0 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %1 = fsm.ref @directory
        %2 = fsm.message @Ack "Put_Ack" %1, %0 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %fwd %2
        fsm.set_delete %cache, %0 : !fsm.set<!fsm.id, 3>, !fsm.id
        %3 = fsm.comp "=" %owner, %0 : !fsm.id, !fsm.id
        fsm.if %3 {
          %4 = fsm.access {memberId = "cl"} %arg0 : !fsm.msg -> !fsm.data
          fsm.update %cl, %4 : !fsm.data, !fsm.data
          %5 = fsm.constant {value = "I"} : !fsm.state
          fsm.update %State, %5 : !fsm.state, !fsm.state
        }
      }
      fsm.transition @PutS(%arg0: !fsm.msg) attributes {nextState = @M} {
        fsm.call @PutM(%arg0)
      }
    }
    fsm.state @M_GetS transitions  {
      fsm.transition @WB(%arg0: !fsm.msg) {
        %0 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %1 = fsm.comp "=" %0, %owner : !fsm.id, !fsm.id
        fsm.if %1 {
          %2 = fsm.access {memberId = "cl"} %arg0 : !fsm.msg -> !fsm.data
          fsm.update %cl, %2 : !fsm.data, !fsm.data
          %3 = fsm.constant {value = "S"} : !fsm.state
          fsm.update %State, %3 : !fsm.state, !fsm.state
        }
      }
    }
  }
}

