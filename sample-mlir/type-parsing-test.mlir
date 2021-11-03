module  {
  %0 = pcc.constant {id = "NrCaches", val = 3 : i64} : i64
  %1 = pcc.net_decl {netId = "fwd"} : !pcc.network<ordered>
  %2 = pcc.net_decl {netId = "resp"} : !pcc.network<unordered>
  %3 = pcc.net_decl {netId = "req"} : !pcc.network<unordered>
  %4 = pcc.cache_decl {State = !pcc.state<I>, cl = !pcc.data, id = "cache"} : !pcc.set<!pcc.struct<!pcc.state<I>, !pcc.data>, 3>
  %5 = pcc.directory_decl {State = !pcc.state<I>, cl = !pcc.data, id = "directory", owner = !pcc.id} : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id>
  %6 = pcc.msg_decl {dst = !pcc.id, id = "Request", mtype = !pcc.mtype<none>, src = !pcc.id} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
  %7 = pcc.msg_decl {dst = !pcc.id, id = "Ack", mtype = !pcc.mtype<none>, src = !pcc.id} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
  %8 = pcc.msg_decl {cl = !pcc.data, dst = !pcc.id, id = "Resp", mtype = !pcc.mtype<none>, src = !pcc.id} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
  %9 = pcc.msg_decl {cl = !pcc.data, dst = !pcc.id, id = "RespAck", mtype = !pcc.mtype<none>, src = !pcc.id} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
  pcc.process @cache_I_store(%arg0: !pcc.struct<!pcc.state<I>, !pcc.data>) attributes {action = "store", start_state = "cache_I"} {
    %10 = pcc.struct_access %arg0["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data> -> !pcc.id
    %11 = pcc.struct_access %5["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> -> !pcc.id
    %12 = pcc.msg_constr %10 %11 {mtype = !pcc.mtype<GetM>} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
    pcc.send : %3 : !pcc.network<unordered> %12 : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
    pcc.await :  {
      pcc.when [ GetM_Ack_D ] (%arg1: !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>) {
        %13 = pcc.struct_access %arg1["cl"] : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data> -> !pcc.data
        pcc.update [cl] %arg0:!pcc.struct<!pcc.state<I>, !pcc.data> %13:!pcc.data
        %14 = pcc.inl_const #pcc.state_attr<M>
        pcc.update [State] %arg0:!pcc.struct<!pcc.state<I>, !pcc.data> %14:!pcc.state<M>
        pcc.break
      }
    }
  }
  pcc.process @cache_I_load(%arg0: !pcc.struct<!pcc.state<I>, !pcc.data>) attributes {action = "load", start_state = "cache_I"} {
    %10 = pcc.struct_access %arg0["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data> -> !pcc.id
    %11 = pcc.struct_access %5["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> -> !pcc.id
    %12 = pcc.msg_constr %10 %11 {mtype = !pcc.mtype<GetM>} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
    pcc.send : %3 : !pcc.network<unordered> %12 : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
    pcc.await :  {
      pcc.when [ GetM_Ack_D ] (%arg1: !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>) {
        %13 = pcc.struct_access %arg1["cl"] : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data> -> !pcc.data
        pcc.update [cl] %arg0:!pcc.struct<!pcc.state<I>, !pcc.data> %13:!pcc.data
        %14 = pcc.inl_const #pcc.state_attr<M>
        pcc.update [State] %arg0:!pcc.struct<!pcc.state<I>, !pcc.data> %14:!pcc.state<M>
        pcc.break
      }
    }
  }
  pcc.process @cache_M_load(%arg0: !pcc.struct<!pcc.state<I>, !pcc.data>) attributes {action = "load", end_state = "M", start_state = "cache_M"} {
  }
  pcc.process @cache_M_store(%arg0: !pcc.struct<!pcc.state<I>, !pcc.data>) attributes {action = "store", end_state = "M", start_state = "cache_M"} {
  }
  pcc.process @cache_M_Fwd_GetM(%arg0: !pcc.struct<!pcc.state<I>, !pcc.data>, %arg1: !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>) attributes {action = "Fwd_GetM", end_state = "I", start_state = "cache_M"} {
    %10 = pcc.struct_access %arg0["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data> -> !pcc.id
    %11 = pcc.struct_access %arg1["src"] : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id> -> !pcc.id
    %12 = pcc.msg_constr %10 %11 {mtype = !pcc.mtype<GetM_Ack_D>} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
    pcc.send : %2 : !pcc.network<unordered> %12 : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
  }
  pcc.process @cache_M_evict(%arg0: !pcc.struct<!pcc.state<I>, !pcc.data>) attributes {action = "evict", start_state = "cache_M"} {
    %10 = pcc.struct_access %arg0["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data> -> !pcc.id
    %11 = pcc.struct_access %5["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> -> !pcc.id
    %12 = pcc.msg_constr %10 %11 {mtype = !pcc.mtype<PutM>} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
    pcc.send : %3 : !pcc.network<unordered> %12 : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
    pcc.await :  {
      pcc.when [ Put_Ack ] (%arg1: !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>) {
        %13 = pcc.inl_const #pcc.state_attr<I>
        pcc.update [State] %arg0:!pcc.struct<!pcc.state<I>, !pcc.data> %13:!pcc.state<I>
        pcc.break
      }
    }
  }
  pcc.process @directory_I_GetM(%arg0: !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id>, %arg1: !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>) attributes {action = "GetM", end_state = "M", start_state = "directory_I"} {
    %10 = pcc.struct_access %arg0["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> -> !pcc.id
    %11 = pcc.struct_access %arg1["src"] : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id> -> !pcc.id
    %12 = pcc.msg_constr %10 %11 {mtype = !pcc.mtype<GetM_Ack_D>} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
    pcc.send : %2 : !pcc.network<unordered> %12 : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
    %13 = pcc.struct_access %arg1["src"] : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id> -> !pcc.id
    pcc.update [owner] %arg0:!pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> %13:!pcc.id
  }
  pcc.process @directory_M_GetM(%arg0: !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id>, %arg1: !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>) attributes {action = "GetM", end_state = "M", start_state = "directory_M"} {
    %10 = pcc.struct_access %arg1["src"] : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id> -> !pcc.id
    %11 = pcc.struct_access %arg0["owner"] : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> -> !pcc.id
    %12 = pcc.msg_constr %10 %11 {mtype = !pcc.mtype<Fwd_GetM>} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
    pcc.send : %1 : !pcc.network<ordered> %12 : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
    %13 = pcc.struct_access %arg1["src"] : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id> -> !pcc.id
    pcc.update [owner] %arg0:!pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> %13:!pcc.id
  }
  pcc.process @directory_M_PutM(%arg0: !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id>, %arg1: !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>) attributes {action = "PutM", start_state = "directory_M"} {
    %10 = pcc.struct_access %arg0["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> -> !pcc.id
    %11 = pcc.struct_access %arg1["src"] : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data> -> !pcc.id
    %12 = pcc.msg_constr %10 %11 {mtype = !pcc.mtype<Put_Ack>} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
    pcc.send : %1 : !pcc.network<ordered> %12 : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
    %true = constant true
    pcc.if %true {
      %13 = pcc.struct_access %arg1["cl"] : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data> -> !pcc.data
      pcc.update [cl] %arg0:!pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> %13:!pcc.data
      %14 = pcc.inl_const #pcc.state_attr<I>
      pcc.update [State] %arg0:!pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> %14:!pcc.state<I>
    }
  }
}