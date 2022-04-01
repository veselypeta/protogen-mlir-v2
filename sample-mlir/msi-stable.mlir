module  {
  %0 = fsm.constant {value = 3 : i64} : i64
  %fwd = fsm.network @fwd "ordered"
  %resp = fsm.network @resp "unordered"
  %req = fsm.network @req "unordered"
  fsm.machine @cache() {
    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data
    %acksReceived = fsm.variable "acksReceived" {initValue = 0 : i64} : !fsm.range<0, 3>
    %acksExpected = fsm.variable "acksExpected" {initValue = 0 : i64} : !fsm.range<0, 3>
    fsm.state @I transitions  {
      fsm.transition @load() {
        %1 = fsm.ref @cache
        %2 = fsm.ref @directory
        %3 = fsm.message @Request "GetS" %1, %2 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %req %3
        fsm.await actions  {
          fsm.when @GetS_Ack(%arg0: !fsm.msg) {
            %4 = fsm.access {memberId = "cl"} %arg0 : !fsm.msg -> !fsm.data
            fsm.update %cl, %4 : !fsm.data, !fsm.data
            %5 = fsm.constant {value = "S"} : !fsm.state
            fsm.update %State, %5 : !fsm.state, !fsm.state
            fsm.break
          }
        }
      }
      fsm.transition @store() {
        %1 = fsm.ref @cache
        %2 = fsm.ref @directory
        %3 = fsm.message @Request "GetM" %1, %2 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %req %3
        %4 = fsm.constant {value = 0 : i64} : i64
        fsm.update %acksReceived, %4 : !fsm.range<0, 3>, i64
        fsm.await actions  {
          fsm.when @GetM_Ack_D(%arg0: !fsm.msg) {
            %5 = fsm.access {memberId = "cl"} %arg0 : !fsm.msg -> !fsm.data
            fsm.update %cl, %5 : !fsm.data, !fsm.data
            %6 = fsm.constant {value = "M"} : !fsm.state
            fsm.update %State, %6 : !fsm.state, !fsm.state
            fsm.break
          }
          fsm.when @GetM_Ack_AD(%arg0: !fsm.msg) {
            %5 = fsm.access {memberId = "acksExpected"} %arg0 : !fsm.msg -> !fsm.range<0, 3>
            fsm.update %acksExpected, %5 : !fsm.range<0, 3>, !fsm.range<0, 3>
            %6 = fsm.comp "==" %acksExpected, %acksReceived : !fsm.range<0, 3>, !fsm.range<0, 3>
            fsm.if %6 {
              %7 = fsm.constant {value = "M"} : !fsm.state
              fsm.update %State, %7 : !fsm.state, !fsm.state
            }
            fsm.await actions  {
              fsm.when @Inv_Ack(%arg1: !fsm.msg) {
                %7 = fsm.constant {value = 1 : i64} : i64
                %8 = fsm.add %acksReceived, %7 : !fsm.range<0, 3>, i64
                fsm.update %acksReceived, %8 : !fsm.range<0, 3>, i64
                %9 = fsm.comp "==" %acksExpected, %acksReceived : !fsm.range<0, 3>, !fsm.range<0, 3>
                fsm.if %9 {
                  %10 = fsm.constant {value = "M"} : !fsm.state
                  fsm.update %State, %10 : !fsm.state, !fsm.state
                }
              }
            }
          }
          fsm.when @Inv_Ack(%arg0: !fsm.msg) {
            %5 = fsm.constant {value = 1 : i64} : i64
            %6 = fsm.add %acksReceived, %5 : !fsm.range<0, 3>, i64
            fsm.update %acksReceived, %6 : !fsm.range<0, 3>, i64
          }
        }
      }
    }
    fsm.state @S transitions  {
      fsm.transition @load() attributes {nextState = @S} {
        fsm.nop
      }
      fsm.transition @store() {
        %1 = fsm.ref @cache
        %2 = fsm.ref @directory
        %3 = fsm.message @Request "Upgrade" %1, %2 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %req %3
        %4 = fsm.constant {value = 0 : i64} : i64
        fsm.update %acksReceived, %4 : !fsm.range<0, 3>, i64
        fsm.await actions  {
          fsm.when @GetM_Ack_D(%arg0: !fsm.msg) {
            %5 = fsm.constant {value = "M"} : !fsm.state
            fsm.update %State, %5 : !fsm.state, !fsm.state
            fsm.break
          }
          fsm.when @GetM_Ack_AD(%arg0: !fsm.msg) {
            %5 = fsm.access {memberId = "acksExpected"} %arg0 : !fsm.msg -> !fsm.range<0, 3>
            fsm.update %acksExpected, %5 : !fsm.range<0, 3>, !fsm.range<0, 3>
            %6 = fsm.comp "==" %acksExpected, %acksReceived : !fsm.range<0, 3>, !fsm.range<0, 3>
            fsm.if %6 {
              %7 = fsm.constant {value = "M"} : !fsm.state
              fsm.update %State, %7 : !fsm.state, !fsm.state
            }
            fsm.await actions  {
              fsm.when @Inv_Ack(%arg1: !fsm.msg) {
                %7 = fsm.constant {value = 1 : i64} : i64
                %8 = fsm.add %acksReceived, %7 : !fsm.range<0, 3>, i64
                fsm.update %acksReceived, %8 : !fsm.range<0, 3>, i64
                %9 = fsm.comp "==" %acksExpected, %acksReceived : !fsm.range<0, 3>, !fsm.range<0, 3>
                fsm.if %9 {
                  %10 = fsm.constant {value = "M"} : !fsm.state
                  fsm.update %State, %10 : !fsm.state, !fsm.state
                }
              }
            }
          }
          fsm.when @Inv_Ack(%arg0: !fsm.msg) {
            %5 = fsm.constant {value = 1 : i64} : i64
            %6 = fsm.add %acksReceived, %5 : !fsm.range<0, 3>, i64
            fsm.update %acksReceived, %6 : !fsm.range<0, 3>, i64
          }
        }
      }
      fsm.transition @evict() {
        %1 = fsm.ref @cache
        %2 = fsm.ref @directory
        %3 = fsm.message @Request "PutS" %1, %2 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %req %3
        fsm.await actions  {
          fsm.when @Put_Ack(%arg0: !fsm.msg) {
            %4 = fsm.constant {value = "I"} : !fsm.state
            fsm.update %State, %4 : !fsm.state, !fsm.state
            fsm.break
          }
        }
      }
      fsm.transition @Inv(%arg0: !fsm.msg) attributes {nextState = @I} {
        %1 = fsm.ref @cache
        %2 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %3 = fsm.message @Resp "Inv_Ack" %1, %2, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %3
      }
    }
    fsm.state @M transitions  {
      fsm.transition @load() {
        fsm.nop
      }
      fsm.transition @store() attributes {nextState = @M} {
        fsm.nop
      }
      fsm.transition @Fwd_GetM(%arg0: !fsm.msg) attributes {nextState = @I} {
        %1 = fsm.ref @cache
        %2 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %3 = fsm.message @Resp "GetM_Ack_D" %1, %2, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %3
      }
      fsm.transition @Fwd_GetS(%arg0: !fsm.msg) attributes {nextState = @S} {
        %1 = fsm.ref @cache
        %2 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %3 = fsm.message @Resp "GetS_Ack" %1, %2, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %3
        %4 = fsm.ref @cache
        %5 = fsm.ref @directory
        %6 = fsm.message @Resp "WB" %4, %5, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %6
      }
      fsm.transition @evict() {
        %1 = fsm.ref @cache
        %2 = fsm.ref @directory
        %3 = fsm.message @Resp "PutM" %1, %2, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %req %3
        fsm.await actions  {
          fsm.when @Put_Ack(%arg0: !fsm.msg) {
            %4 = fsm.constant {value = "I"} : !fsm.state
            fsm.update %State, %4 : !fsm.state, !fsm.state
            fsm.break
          }
        }
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
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.set_add %cache, %1 : !fsm.set<!fsm.id, 3>, !fsm.id
        %2 = fsm.ref @directory
        %3 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %4 = fsm.message @Resp "GetS_Ack" %2, %3, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %4
      }
      fsm.transition @GetM(%arg0: !fsm.msg) attributes {nextState = @M} {
        %1 = fsm.ref @directory
        %2 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %3 = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
        %4 = fsm.message @RespAck "GetM_Ack_AD" %1, %2, %cl, %3 : !fsm.id, !fsm.id, !fsm.data, i64 -> !fsm.msg
        fsm.send %resp %4
        %5 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.update %owner, %5 : !fsm.id, !fsm.id
      }
    }
    fsm.state @S transitions  {
      fsm.transition @GetS(%arg0: !fsm.msg) {
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.set_add %cache, %1 : !fsm.set<!fsm.id, 3>, !fsm.id
        %2 = fsm.ref @directory
        %3 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %4 = fsm.message @Resp "GetS_Ack" %2, %3, %cl : !fsm.id, !fsm.id, !fsm.data -> !fsm.msg
        fsm.send %resp %4
      }
      fsm.transition @Upgrade(%arg0: !fsm.msg) {
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.set_contains %cache, %1 : !fsm.set<!fsm.id, 3>, !fsm.id
        fsm.if %2 {
          %7 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
          fsm.set_delete %cache, %7 : !fsm.set<!fsm.id, 3>, !fsm.id
          %8 = fsm.ref @directory
          %9 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
          %10 = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
          %11 = fsm.message @RespAck "GetM_Ack_AD" %8, %9, %cl, %10 : !fsm.id, !fsm.id, !fsm.data, i64 -> !fsm.msg
          fsm.send %resp %11
          %12 = fsm.constant {value = "M"} : !fsm.state
          fsm.update %State, %12 : !fsm.state, !fsm.state
        } else {
          %7 = fsm.ref @directory
          %8 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
          %9 = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
          %10 = fsm.message @RespAck "GetM_Ack_AD" %7, %8, %cl, %9 : !fsm.id, !fsm.id, !fsm.data, i64 -> !fsm.msg
          fsm.send %resp %10
          %11 = fsm.constant {value = "M"} : !fsm.state
          fsm.update %State, %11 : !fsm.state, !fsm.state
        }
        %3 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %4 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %5 = fsm.message @Ack "Inv" %3, %4 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.multicast %fwd %5 %cache : !fsm.set<!fsm.id, 3>
        %6 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.update %owner, %6 : !fsm.id, !fsm.id
        fsm.set_clear %cache : !fsm.set<!fsm.id, 3>
      }
      fsm.transition @PutS(%arg0: !fsm.msg) {
        %1 = fsm.ref @directory
        %2 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %3 = fsm.message @Ack "Put_Ack" %1, %2 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %fwd %3
        %4 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.set_delete %cache, %4 : !fsm.set<!fsm.id, 3>, !fsm.id
        %5 = fsm.set_count %cache : !fsm.set<!fsm.id, 3>
        %6 = fsm.constant {value = 0 : i64} : i64
        %7 = fsm.comp "==" %5, %6 : i64, i64
        fsm.if %7 {
          %8 = fsm.constant {value = "I"} : !fsm.state
          fsm.update %State, %8 : !fsm.state, !fsm.state
        }
      }
    }
    fsm.state @M transitions  {
      fsm.transition @GetS(%arg0: !fsm.msg) {
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.message @Request "Fwd_GetS" %1, %owner : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %fwd %2
        %3 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.set_add %cache, %3 : !fsm.set<!fsm.id, 3>, !fsm.id
        fsm.set_add %cache, %owner : !fsm.set<!fsm.id, 3>, !fsm.id
        fsm.await actions  {
          fsm.when @WB(%arg1: !fsm.msg) {
            %4 = fsm.access {memberId = "src"} %arg1 : !fsm.msg -> !fsm.id
            %5 = fsm.comp "==" %4, %owner : !fsm.id, !fsm.id
            fsm.if %5 {
              %6 = fsm.access {memberId = "cl"} %arg1 : !fsm.msg -> !fsm.data
              fsm.update %cl, %6 : !fsm.data, !fsm.data
              %7 = fsm.constant {value = "S"} : !fsm.state
              fsm.update %State, %7 : !fsm.state, !fsm.state
            }
          }
        }
      }
      fsm.transition @GetM(%arg0: !fsm.msg) {
        %1 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %2 = fsm.message @Request "Fwd_GetM" %1, %owner : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %fwd %2
        %3 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.update %owner, %3 : !fsm.id, !fsm.id
      }
      fsm.transition @PutM(%arg0: !fsm.msg) {
        %1 = fsm.ref @directory
        %2 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %3 = fsm.message @Ack "Put_Ack" %1, %2 : !fsm.id, !fsm.id -> !fsm.msg
        fsm.send %fwd %3
        %4 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        fsm.set_delete %cache, %4 : !fsm.set<!fsm.id, 3>, !fsm.id
        %5 = fsm.access {memberId = "src"} %arg0 : !fsm.msg -> !fsm.id
        %6 = fsm.comp "==" %owner, %5 : !fsm.id, !fsm.id
        fsm.if %6 {
          %7 = fsm.access {memberId = "cl"} %arg0 : !fsm.msg -> !fsm.data
          fsm.update %cl, %7 : !fsm.data, !fsm.data
          %8 = fsm.constant {value = "I"} : !fsm.state
          fsm.update %State, %8 : !fsm.state, !fsm.state
        }
      }
    }
  }
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
}
