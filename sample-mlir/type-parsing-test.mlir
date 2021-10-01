module  {
  %0 = pcc.constant {id = "NrCaches", val = 3 : i64} : i64
  %1 = pcc.net_decl {netId = "fwd"} : !pcc.network<ordered>
  %2 = pcc.net_decl {netId = "resp"} : !pcc.network<unordered>
  %3 = pcc.net_decl {netId = "req"} : !pcc.network<unordered>
  %4 = pcc.cache_decl {State = !pcc.state<I>, acksExpected = !pcc.int_range<0, 3>, acksReceived = !pcc.int_range<0, 3>, cl = !pcc.data, id = "cache"} : !pcc.set<!pcc.struct<!pcc.state<I>, !pcc.data, !pcc.int_range<0, 3>, !pcc.int_range<0, 3>>, 3>
  %5 = pcc.directory_decl {State = !pcc.state<I>, cache = !pcc.set<!pcc.id, 3>, cl = !pcc.data, id = "directory", owner = !pcc.id} : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.set<!pcc.id, 3>, !pcc.id>
  %6 = pcc.msg_decl {dst = !pcc.id, id = "Request", mtype = !pcc.mtype<none>, src = !pcc.id} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
  %7 = pcc.msg_decl {dst = !pcc.id, id = "Ack", mtype = !pcc.mtype<none>, src = !pcc.id} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
  %8 = pcc.msg_decl {cl = !pcc.data, dst = !pcc.id, id = "Resp", mtype = !pcc.mtype<none>, src = !pcc.id} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
  %9 = pcc.msg_decl {acksExpected = !pcc.int_range<0, 3>, cl = !pcc.data, dst = !pcc.id, id = "RespAck", mtype = !pcc.mtype<none>, src = !pcc.id} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data, !pcc.int_range<0, 3>>
  pcc.process @cache_I_load(%arg0: !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.int_range<0, 3>, !pcc.int_range<0, 3>>) {
    %10 = pcc.struct_access %arg0["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.int_range<0, 3>, !pcc.int_range<0, 3>> -> !pcc.id
    %11 = pcc.struct_access %5["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.set<!pcc.id, 3>, !pcc.id> -> !pcc.id
    %12 = pcc.msg_constr %10 %11 {mtype = !pcc.mtype<GetS>} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
    pcc.await :  {
      pcc.when [ GetS ] (%arg1: !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>) {
        pcc.break
      }
    }
  }
}