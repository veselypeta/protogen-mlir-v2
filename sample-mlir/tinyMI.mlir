module {
      %0 = pcc.constant {id = "NrCaches", val = 3 : i64} : i64

      %1 = pcc.net_decl {netId = "fwd"} : !pcc.network<ordered>
      %2 = pcc.net_decl {netId = "resp"} : !pcc.network<unordered>
      %3 = pcc.net_decl {netId = "req"} : !pcc.network<unordered>


      %4 = pcc.cache_decl {State = !pcc.state<I>, acksExpected = !pcc.int_range<0, 3>, acksReceived = !pcc.int_range<0, 3>, cl = !pcc.data, id = "cache"} : !pcc.set<!pcc.struct<!pcc.state<I>, !pcc.data, !pcc.int_range<0, 3>, !pcc.int_range<0, 3>>, 3>
      %5 = pcc.directory_decl {State = !pcc.state<I>, cache = !pcc.set<!pcc.id, 3>, cl = !pcc.data, id = "directory", owner = !pcc.id} : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id, !pcc.set<!pcc.id, 3>>

      %6 = pcc.msg_decl {dst = !pcc.id, id = "Request", mtype = !pcc.mtype<none>, src = !pcc.id} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>
      %7 = pcc.msg_decl {cl = !pcc.data, dst = !pcc.id, id = "Resp", mtype = !pcc.mtype<none>, src = !pcc.id} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
    }