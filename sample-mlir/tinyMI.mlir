module {
      %0 = pcc.constant {id = "NrCaches", val = 3 : i64} : i64
      %1 = pcc.net_decl {netId = "fwd"} : !pcc.network<ordered>
      %2 = pcc.net_decl {netId = "resp"} : !pcc.network<unordered>
      %3 = pcc.net_decl {netId = "req"} : !pcc.network<unordered>
      %4 = pcc.cache_decl {State = !pcc.state<I>, cl = !pcc.data, id = "cache"} : !pcc.set<!pcc.struct<!pcc.state<I>, !pcc.data>, 3>
      %5 = pcc.directory_decl {State = !pcc.state<I>, cl = !pcc.data, id = "directory", owner = !pcc.id} : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id>
}